{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE TypeApplications, PartialTypeSignatures #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ConstraintKinds #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module Main where

import           Control.Applicative
import           Control.Category
import           Control.Concurrent
import           Control.DeepSeq
import           Control.Monad
import           Control.Monad.Trans
import           Control.Monad.IO.Class
import qualified Control.Monad.State.Lazy as SL
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as LB
import qualified Data.ByteString.Lazy.Char8 as LC
import           Data.Char
import           Data.Foldable
import           Data.Functor.Identity
import           Data.IORef
import           Data.List
import           Data.Maybe
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Random.Distribution.Categorical
import           Data.Traversable
import           Data.Type.Equality ((:~:)(..))
import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
import           Data.Word
import           Debug.Trace
import           GHC.TypeLits
import           Options.Applicative
import           System.Clock
import           System.Environment
import           System.IO
import           System.IO.Unsafe
import           System.Random
import           Text.Printf

import           AI.Funn.CL.Blob (Blob(..))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.Flat
import           AI.Funn.CL.LSTM
import           AI.Funn.CL.Mixing
import           AI.Funn.CL.MonadCL
import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..), (>>>))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Diff.Pointed
import           AI.Funn.Diff.RNN
import qualified AI.Funn.Flat.Blob as F
import           AI.Funn.Flat.ParBox
import           AI.Funn.SGD
import           AI.Funn.Space
import           AI.Funn.TypeLits

sampleIO :: MonadIO m => RVar a -> m a
sampleIO v = liftIO (runRVar v StdRandom)

deepseqM :: (Monad m, NFData a) => a -> m ()
deepseqM x = deepseq x (return ())

unfoldM :: Monad m => Int -> m a -> m [a]
unfoldM 0 m = return []
unfoldM n m = (:) <$> m <*> unfoldM (n-1) m

onehot :: Vector (Blob Double 256)
onehot = unsafePerformIO $ sequenceA $ V.generate 256 gen
  where
    gen i = Blob.fromList (replicate i 0 ++ [1] ++ replicate (255 - i) 0)

fromGPU :: (MonadIO m, KnownNat n) => Blob Double n -> m (F.Blob n)
fromGPU b = F.fromList <$> Blob.toList b

toGPU :: (MonadIO m, KnownNat n) => F.Blob n -> m (Blob Double n)
toGPU b = Blob.fromList (F.toList b)

sampleRNN :: (MonadIO m) => Int -> st -> (st -> Blob Double 256 -> m (st, Blob Double 256)) -> m [Char]
sampleRNN n s0 next = SL.evalStateT (unfoldM n step) (s0, 0)
  where
    step = do
      (s, c) <- SL.get
      (new_s, new_q) <- lift $ next s (onehot V.! c)
      new_qc <- fromGPU new_q
      let
        exps = V.map exp (F.getBlob new_qc)
        factor = 1 / V.sum exps
        ps = V.map (*factor) exps
      new_c <- sampleIO (categorical $ zip (V.toList ps) [0 :: Int ..])
      SL.put (new_s, new_c)
      return (chr new_c)

data Commands = Train (Maybe FilePath) FilePath (Maybe FilePath) (Maybe FilePath) Int Double Integer
              | Sample FilePath (Maybe Int)
              deriving (Show)

instance (Additive m a, Applicative m) => Zero m (Vector a) where
  zero = pure V.empty

instance (Additive m a, Monad m) => Semi m (Vector a) where
  plus xs ys
    | V.null xs = pure ys
    | V.null ys = pure xs
    | otherwise = V.zipWithM plus xs ys

instance (Additive m a, Monad m) => Additive m (Vector a)

type BlobD = Blob Double

lstm :: (MonadIO m, KnownNat n)
      => Ref s (BlobD (2*n))
      -> Ref s (BlobD n)
      -> Ref s (BlobD (4*n))
      -> Pointed m s (Ref s (BlobD n), Ref s (BlobD n))
lstm pars hidden inputs = do ins <- packrec (pars, (hidden, inputs))
                             out <- lstmDiff <-- ins
                             unpack out

amix :: (MonadIO m, KnownNat size, KnownNat a, KnownNat b)
     => Proxy size -> Ref s (BlobD _) -> Ref s (BlobD a)
     -> Pointed m s (Ref s (BlobD b))
amix p pars input = do ins <- pack (pars, input)
                       pushDiff ins (amixDiff p)

tanhP :: (MonadIO m, KnownNat a) => Ref s (BlobD a) -> Pointed m s (Ref s (BlobD a))
tanhP input = pushDiff input tanhDiff

split3 :: (MonadIO m, KnownNat a, KnownNat b, KnownNat c)
       => Ref s (BlobD (a+b+c))
       -> Pointed m s (Ref s (BlobD a), Ref s (BlobD b), Ref s (BlobD c))
split3 input = do (part1, part23) <- splitDiff =<- input
                  (part2,  part3) <- splitDiff =<- part23
                  return (part1, part2, part3)

type Affine m a = (Derivable a, Additive m (D a))

scanlP :: (Monad m, Affine m x, Affine m st, Affine m i, Affine m o)
       => Diff m (x,(st,i)) (st, o)
       -> Ref s x
       -> Ref s st
       -> Ref s (Vector i)
       -> Pointed m s (Ref s st, Ref s (Vector o))
scanlP diff x st vi = do ins <- packrec (x, (st, vi))
                         out <- pushDiff ins (scanlDiff diff)
                         unpack out

step :: (MonadIO m, KnownNat modelSize) => Proxy modelSize -> Diff m (BlobD _, ((BlobD modelSize, BlobD modelSize), BlobD 256)) ((BlobD modelSize, BlobD modelSize), BlobD 256)
step Proxy = runPointed $ \in_all -> do
  (pars, ((hidden, prev), char)) <- unpackrec in_all
  (p1, p2, p3) <- split3 pars
  combined_in <- mergeDiff -<= (prev, char)
  lstm_in <- tanhP =<< amix (Proxy @ 5) p1 combined_in
  (hidden_out, lstm_out) <- lstm p2 hidden lstm_in
  final_dist <- amix (Proxy @ 5) p3 lstm_out
  packrec ((hidden_out, lstm_out), final_dist)

network :: (MonadIO m, KnownNat modelSize) => Proxy modelSize -> Diff m (BlobD _, Vector (BlobD 256)) (Vector (BlobD 256))
network modelSize = runPointed $ \in_all -> do
  (pars, inputs) <- unpack in_all
  (step_pars, h0, c0) <- split3 pars
  initial_state <- pack (h0, c0)
  (s, vo) <- scanlP (step modelSize) step_pars initial_state inputs
  return vo

evalNetwork :: (MonadIO m, KnownNat modelSize) => Proxy modelSize -> Diff m (BlobD _, (Vector (BlobD 256), Vector Int)) Double
evalNetwork size = Diff.assocL >>> Diff.first network' >>> zipDiff >>> mapDiff softmaxCost >>> vsumDiff
  where
    network' = network size

train :: KnownNat modelSize => Proxy modelSize -> BlobD _ -> FilePath -> (Maybe FilePath) -> (Maybe FilePath) -> Int -> Double -> IO ()
train modelSize initialParameters input savefile logfile chunkSize learningRate =
      do
        let
          step' = step modelSize
          evalNetwork' = evalNetwork modelSize

          runrnn par s c = do
            ((c', o), _) <- Diff.runDiff step' (par, (s, c))
            return (c', o)

        print (natVal initialParameters)

        text <- B.readFile input
        running_average <- newIORef 0
        running_count <- newIORef 0
        iteration <- newIORef (0 :: Int)
        startTime <- getTime ProcessCPUTime
        let
          α :: Double
          α = 0.99

          tvec = V.fromList (B.unpack text) :: U.Vector Word8
          ovec = V.map (\c -> onehot V.! (fromIntegral c)) (V.convert tvec) :: Vector (BlobD 256)

          source :: IO (Vector (BlobD 256), Vector Int)
          source = do s <- sampleIO (uniform 0 (V.length tvec - chunkSize))
                      let
                        input = (onehot V.! 0) `V.cons` V.slice s (chunkSize - 1) ovec
                        output = V.map fromIntegral (V.convert (V.slice s chunkSize tvec))
                      return (input, output)

          objective p = do
            sample <- source
            (err, k) <- Diff.runDiff evalNetwork' (p,sample)

            when (not (isInfinite err || isNaN err)) $ do
              modifyIORef' running_average (\x -> (α*x + (1 - α)*err))
              modifyIORef' running_count (\x -> (α*x + (1 - α)*1))

            -- putStrLn $ "Error: " ++ show err
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
              let (par, c0) = Blob.splitBlob p
              msg <- sampleRNN 200 (Blob.splitBlob c0) (runrnn par)
              putStrLn (filter (\c -> isPrint c || isSpace c) msg)

            when (i `mod` 100 == 0) $ do
              case savefile of
                Just savefile -> do
                  sp <- fromGPU p
                  let encoded = encodeToByteString (natVal modelSize, ParBox sp)
                  LB.writeFile (printf "%s-%6.6i-%5.5f.bin" savefile i x) $ encoded
                  LB.writeFile (savefile ++ "-latest.bin") $ encoded
                Nothing -> return ()
            m

        liftIO $ putStrLn "before adam"

        adam (Blob.adamBlob { adam_α = learningRate }) initialParameters objective next

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
                          (progDesc "Sample output.")))
                   fullDesc)

  cmd <- customExecParser (prefs showHelpOnError) optparser

  initOpenCL

  case cmd of
    Train Nothing input savefile logfile chunkSize lr modelSize -> do
      withNat modelSize $ \(proxy :: Proxy modelSize) -> do
        initial_spar <- sampleIO (F.generate $ uniform (-0.5) (0.5))
        initial_par <- toGPU initial_spar
        train proxy initial_par input savefile logfile chunkSize lr

    Train (Just resumepath) input savefile logfile chunkSize lr _ -> do
      (modelSize, box) <- decodeOrError <$> LB.readFile resumepath
      withNat modelSize $ \(proxy :: Proxy modelSize) ->
        case openParBox box of
          Just initial_spar -> do
            initial_par <- toGPU initial_spar
            train proxy initial_par input savefile logfile chunkSize lr
          Nothing -> error "model mismatch"

    Sample initpath length -> do
      (modelSize, box) <- decodeOrError <$> LB.readFile initpath
      let n = fromMaybe 500 length
      withNat modelSize $ \(proxy :: Proxy modelSize) ->
        case openParBox box of
          Just sinitial -> do
            initial <- toGPU sinitial
            let
              (par, c0) = Blob.splitBlob initial
              runrnn s c = do
                ((c', o), _) <- Diff.runDiff (step proxy) (par, (s, c))
                return (c', o)
            msg <- sampleRNN n (Blob.splitBlob c0) runrnn
            putStrLn msg
          Nothing -> error "model mismatch"
