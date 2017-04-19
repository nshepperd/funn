
{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, ForeignFunctionInterface #-}
{-# LANGUAGE TypeApplications, PartialTypeSignatures #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
module Main where
import           Control.Applicative
import           Control.Monad
import           Control.Category
import           Data.Foldable
import           Data.Monoid
import           Data.Traversable

import           Data.Char
import           Data.IORef
import           Data.List
import           Data.Maybe
import           Data.Proxy
import           Data.Word

import           Control.Concurrent
import           System.Clock
import           System.Environment
import           System.IO

import           Options.Applicative

import           Text.Printf

import qualified Data.Binary as LB
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as LB
import qualified Data.ByteString.Lazy.Char8 as LC

import           Control.DeepSeq
import           Data.Coerce
import           Data.Type.Equality
import           Debug.Trace
import           GHC.TypeLits

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe

import qualified Control.Monad.State.Lazy as SL
import           Control.Monad.IO.Class
import           Data.Functor.Identity
import           Data.Random
import           Data.Random.Distribution.Categorical
import           System.Random

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M
import qualified Numeric.LinearAlgebra.HMatrix as HM

import           AI.Funn.Common
import           AI.Funn.Flat.Blob (Blob(..))
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Diff.Diff (Diff(..), Additive(..), Derivable(..), (>>>))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Network.Network
import           AI.Funn.Network.Flat
import           AI.Funn.Network.RNN
import           AI.Funn.Network.LSTM
import           AI.Funn.Network.Mixing
import           AI.Funn.SGD
import           AI.Funn.SomeNat

sampleIO :: MonadIO m => RVar a -> m a
sampleIO v = liftIO (runRVar v StdRandom)

deepseqM :: (Monad m, NFData a) => a -> m ()
deepseqM x = deepseq x (return ())

average :: [Double] -> Double
average xs = sum xs / genericLength xs
stdev :: [Double] -> Double
stdev xs = let m = average xs in
            sum [(x-m)^2 | x <- xs] / (genericLength xs - 1)

unfoldM :: Monad m => Int -> m a -> m [a]
unfoldM 0 m = return []
unfoldM n m = (:) <$> m <*> unfoldM (n-1) m

onehot :: Vector (Blob 128)
onehot = V.generate 128 (\i -> Blob.fromList (replicate i 0 ++ [1] ++ replicate (127 - i) 0))

sampleRNN :: Int -> s -> (s -> Blob 128 -> IO (s, Blob 128)) -> IO [Char]
sampleRNN n s0 next = SL.evalStateT (unfoldM n step) (s0, 0)
  where
    step = do
      (s, c) <- SL.get
      (new_s, new_q) <- liftIO $ next s (onehot V.! c)
      let exps = V.map exp (getBlob new_q)
          factor = 1 / V.sum exps
          ps = V.map (*factor) exps
      new_c <- sampleIO (categorical $ zip (V.toList ps) [0 :: Int ..])
      SL.put (new_s, new_c)
      return (chr new_c)

data Commands = Train (Maybe FilePath) FilePath (Maybe FilePath) (Maybe FilePath) Int Double Integer
              | Sample FilePath (Maybe Int)
              | CheckDeriv
              deriving (Show)

instance (Additive m a, Monad m) => Additive m (Vector a) where
  zero = pure V.empty
  plus xs ys
    | V.null xs = pure ys
    | V.null ys = pure xs
    | otherwise = V.zipWithM plus xs ys

data ParBox where
  ParBox :: KnownNat n => Blob n -> ParBox

instance LB.Binary (Blob n) where
  put (Blob xs) = putVector putDouble xs
  get = Blob <$> getVector getDouble

instance LB.Binary ParBox where
  put (ParBox (b :: Blob n)) = do
    LB.put (natVal (Proxy @ n))
    LB.put b
  get = do
    n <- LB.get
    withNat n $ \(Proxy :: Proxy n) -> do
      (b :: Blob n) <- LB.get
      return (ParBox b)

openParBox :: forall n. KnownNat n => ParBox -> Maybe (Blob n)
openParBox (ParBox (b :: Blob m)) =
  case sameNat (Proxy @ n) (Proxy @ m) of
    Just Refl -> Just b
    Nothing -> Nothing

step :: (Monad m, KnownNat modelSize) => Proxy modelSize -> Network m ((Blob modelSize, Blob modelSize), Blob 128) ((Blob modelSize, Blob modelSize), Blob 128)
step Proxy = assocR
             >>> second (mergeLayer >>> amixLayer (Proxy @ 5) >>> tanhLayer) -- (Blob n, Blob 4n)
             >>> lstmLayer                                                   -- (Blob n, Blob n)
             >>> second dupLayer >>> assocL >>> second (amixLayer (Proxy @ 5))

network :: (Monad m, KnownNat modelSize) => Proxy modelSize -> Network m (Vector (Blob 128)) (Vector (Blob 128))
network modelSize = fromParams (Blob.generate (pure 0)) -- (Blob 2n, Vector (Blob 128))
                    >>> first splitLayer                -- ((Blob n, Blob n), Vector (Blob 128))
                    >>> scanlLayer (step modelSize)     -- ((Blob n, Blob n), Vector (Blob 128))
                    >>> sndNetwork

evalNetwork :: (Monad m, KnownNat modelSize) => Proxy modelSize -> Network m (Vector (Blob 128), Vector Int) Double
evalNetwork size = first (network size) >>> zipLayer >>> mapLayer softmaxCost >>> vsumLayer

openNetwork :: forall m n a b. KnownNat n => Network m a b -> Maybe (Diff m (Blob n, a) b, RVar (Blob n))
openNetwork (Network p diff initial) =
  case sameNat p (Proxy @ n) of
    Just Refl -> Just (diff, initial)
    Nothing -> Nothing


main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  let optparser = (info (subparser $
                         command "train"
                         (info (Train
                                <$> optional (strOption (long "initial" <> action "file"))
                                <*> strOption (long "input" <> action "file")
                                <*> optional (strOption (long "output" <> action "file"))
                                <*> optional (strOption (long "log" <> action "file"))
                                <*> (option auto (long "chunksize") <|> pure 50)
                                <*> (option auto (long "lr") <|> pure 0.001)
                                <*> (option auto (long "modelSize") <|> pure 200)
                               )
                          (progDesc "Train NN."))
                         <>
                         command "sample"
                         (info (Sample
                                <$> strOption (long "snapshot" <> action "file")
                                <*> optional (option auto (long "length")))
                          (progDesc "Sample output."))
                         <>
                         command "check"
                         (info (pure CheckDeriv)
                          (progDesc "Check Derivatives.")))
                   fullDesc)

  cmd <- customExecParser (prefs showHelpOnError) optparser

  let
    train :: (KnownNat modelSize) => Proxy modelSize -> Maybe ParBox -> FilePath -> (Maybe FilePath) -> (Maybe FilePath) -> Int -> Double -> IO ()
    train modelSize initialParameters input savefile logfile chunkSize learningRate =
      case step modelSize of
        Network pp diff_step init_step ->
          do
            let
              Just (diff_eval, init_eval) = openNetwork (evalNetwork modelSize)
              runrnn par s c = do
                ((c', o), _) <- Diff.runDiff diff_step (par, (s, c))
                return (c', o)

            start_params <- case initialParameters of
                              Just box -> case openParBox box of
                                Just p -> return p
                                Nothing -> error "Model mismatch"
                              Nothing -> sampleIO init_eval

            text <- B.readFile input
            running_average <- newIORef 0
            running_count <- newIORef 0
            iteration <- newIORef (0 :: Int)
            startTime <- getTime ProcessCPUTime
            let
              α :: Double
              α = 0.99

              tvec = V.fromList (filter (<128) $ B.unpack text) :: U.Vector Word8
              ovec = V.map (\c -> onehot V.! (fromIntegral c)) (V.convert tvec) :: Vector (Blob 128)

              source :: IO (Vector (Blob 128), Vector Int)
              source = do s <- sampleIO (uniform 0 (V.length tvec - chunkSize))
                          let
                            input = (onehot V.! 0) `V.cons` V.slice s (chunkSize - 1) ovec
                            output = V.map fromIntegral (V.convert (V.slice s chunkSize tvec))
                          return (input, output)

              objective p = do
                sample <- source
                (err, k) <- Diff.runDiff diff_eval (p,sample)

                when (not (isInfinite err || isNaN err)) $ do
                  modifyIORef' running_average (\x -> (α*x + (1 - α)*err))
                  modifyIORef' running_count (\x -> (α*x + (1 - α)*1))

                putStrLn $ "Error: " ++ show err
                (dp, _) <- k 1
                return dp

              next p m = do
                x <- do q <- readIORef running_average
                        w <- readIORef running_count
                        return ((q / w) / fromIntegral chunkSize)
                modifyIORef' iteration (+1)
                i <- readIORef iteration
                now <- getTime ProcessCPUTime
                let tdiff = fromIntegral (toNanoSecs (now - startTime)) / (10^9) :: Double
                putStrLn $ printf "[% 11.4f | %i]  %f" tdiff i x

                when (i `mod` 50 == 0) $ do
                  let (par, c0) = Blob.split p
                  msg <- sampleRNN 200 (Blob.split c0) (runrnn par)
                  putStrLn msg

                when (i `mod` 100 == 0) $ do
                  case savefile of
                    Just savefile -> do
                      LB.writeFile (printf "%s-%6.6i-%5.5f.bin" savefile i x) $ LB.encode (natVal modelSize, ParBox p)
                      LB.writeFile (savefile ++ "-latest.bin") $ LB.encode (natVal modelSize, ParBox p)
                    Nothing -> return ()
                m

            deepseqM (tvec, ovec)

            adam (Blob.adamBlob { adam_α = learningRate }) start_params objective next


  case cmd of
    Train Nothing input savefile logfile chunkSize lr modelSize -> do
      withNat modelSize $ \(proxy :: Proxy modelSize) -> do
        train proxy Nothing input savefile logfile chunkSize lr

    Train (Just resumepath) input savefile logfile chunkSize lr _ -> do
      (modelSize, box) <- LB.decode <$> LB.readFile resumepath
      withNat modelSize $ \(proxy :: Proxy modelSize) ->
        train proxy box input savefile logfile chunkSize lr

    Sample initpath length -> do
      (modelSize, box) <- LB.decode <$> LB.readFile initpath
      let n = fromMaybe 500 length
      withNat modelSize $ \(proxy :: Proxy modelSize) ->
        case step proxy of
          Network _ diff_step _ ->
            case openParBox box of
              Just initial -> do
                deepseqM initial
                let (par, c0) = Blob.split initial
                    runrnn s c = do
                      ((c', o), _) <- Diff.runDiff diff_step (par, (s, c))
                      return (c', o)
                msg <- sampleRNN n (Blob.split c0) runrnn
                putStrLn msg
              Nothing -> error "model mismatch"
