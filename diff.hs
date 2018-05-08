{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE TypeApplications, PartialTypeSignatures #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ConstraintKinds #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
{-# OPTIONS_GHC -Wno-partial-type-signatures #-}
module Main where

import           Prelude hiding ((.))

import           Control.Applicative
import           Control.Category
import           Control.Concurrent
import           Control.DeepSeq
import           Control.Monad
import           Control.Monad.IO.Class
import qualified Control.Monad.State.Lazy as SL
import           Control.Monad.Trans
import qualified Data.Binary as LB
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
import           System.Random
import           Text.Printf

import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..), (>>>))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Diff.Pointed
import           AI.Funn.Diff.RNN
import           AI.Funn.Flat.Blob (Blob(..), blob, getBlob)
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Flat.Flat
import           AI.Funn.Flat.LSTM
import           AI.Funn.Flat.Mixing
import           AI.Funn.Flat.ParBox
import           AI.Funn.Flat.Softmix
import           AI.Funn.Optimizer.AMSGrad
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Optimizer.SGD
import           AI.Funn.Space
import           AI.Funn.TypeLits

sampleIO :: MonadIO m => RVar a -> m a
sampleIO v = liftIO (runRVar v StdRandom)

deepseqM :: (Monad m, NFData a) => a -> m ()
deepseqM x = deepseq x (return ())

unfoldM :: Monad m => Int -> m a -> m [a]
unfoldM 0 m = return []
unfoldM n m = (:) <$> m <*> unfoldM (n-1) m

onehot :: Vector (Blob 128)
onehot = V.generate 128 (\i -> Blob.fromList (replicate i 0 ++ [1] ++ replicate (127 - i) 0))

sampleRNN :: MonadIO m => Int -> s -> (s -> Blob 128 -> m (s, Blob 128)) -> m [Char]
sampleRNN n s0 next = SL.evalStateT (unfoldM n step) (s0, 0)
  where
    step = do
      (s, c) <- SL.get
      (new_s, new_q) <- lift $ next s (onehot V.! c)
      let exps = V.map exp (getBlob new_q)
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

instance (Additive m a, Monad m) => Additive m (Vector a) where
  {}

lstm :: (Monad m, KnownNat n)
      => Ref s (Blob (2*n))
      -> Ref s (Blob n)
      -> Ref s (Blob (4*n))
      -> Pointed m s (Ref s (Blob n), Ref s (Blob n))
lstm pars hidden inputs = do ins <- packrec (pars, (hidden, inputs))
                             out <- lstmDiff <-- ins
                             unpack out

amix :: (Monad m, KnownNat size, KnownNat a, KnownNat b)
     => Proxy size -> Ref s (Blob _) -> Ref s (Blob a)
     -> Pointed m s (Ref s (Blob b))
amix p pars input = do ins <- pack (pars, input)
                       pushDiff ins (amixDiff p)

tanhP :: (Monad m, KnownNat a) => Ref s (Blob a) -> Pointed m s (Ref s (Blob a))
tanhP input = pushDiff input tanhDiff

split3 :: (Monad m, KnownNat a, KnownNat b, KnownNat c)
       => Ref s (Blob (a+b+c))
       -> Pointed m s (Ref s (Blob a), Ref s (Blob b), Ref s (Blob c))
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

step :: (Monad m, KnownNat modelSize) => Proxy modelSize -> Diff m (Blob _, ((Blob modelSize, Blob modelSize), Blob 128)) ((Blob modelSize, Blob modelSize), Blob 128)
step Proxy = runPointed $ \in_all -> do
  (pars, ((hidden, prev), char)) <- unpackrec in_all
  (p1, p2, p3) <- split3 pars
  combined_in <- mergeDiff -<= (prev, char)
  lstm_in <- tanhP =<< amix (Proxy @ 5) p1 combined_in
  (hidden_out, lstm_out) <- lstm p2 hidden lstm_in
  final_dist <- amix (Proxy @ 5) p3 lstm_out
  packrec ((hidden_out, lstm_out), final_dist)

network :: (Monad m, KnownNat modelSize) => Proxy modelSize -> Diff m (Blob _, Vector (Blob 128)) (Vector (Blob 128))
network modelSize = runPointed $ \in_all -> do
  (pars, inputs) <- unpack in_all
  (step_pars, h0, c0) <- split3 pars
  initial_state <- pack (h0, c0)
  (s, vo) <- scanlP (step modelSize) step_pars initial_state inputs
  return vo

evalNetwork :: (Monad m, KnownNat modelSize) => Proxy modelSize -> Diff m (Blob _, (Vector (Blob 128), Vector Int)) Double
evalNetwork size = Diff.assocL >>> Diff.first network' >>> zipDiff >>> mapDiff softmaxCost >>> vsumDiff
  where
    network' = network size

train :: KnownNat modelSize => Proxy modelSize -> Blob _ -> FilePath -> (Maybe FilePath) -> (Maybe FilePath) -> Int -> Double -> IO ()
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
        (err, k) <- Diff.runDiff evalNetwork' (p,sample)

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
          putStrLn (filter (\c -> isPrint c || isSpace c) msg)

        when (i `mod` 100 == 0) $ do
          case savefile of
            Just savefile -> do
              LB.writeFile (printf "%s-%6.6i-%5.5f.bin" savefile i x) $ LB.encode (natVal modelSize, ParBox p)
              LB.writeFile (savefile ++ "-latest.bin") $ LB.encode (natVal modelSize, ParBox p)
            Nothing -> return ()
        m

    deepseqM (tvec, ovec)

    -- trainState <- initSGD learningRate 0.9 plus initialParameters
    -- let go trainState = do
    --       grad <- objective (extractSGD trainState)
    --       trainState' <- updateSGD grad trainState
    --       next (extractSGD trainState') (go trainState')

    trainState <- initAdam learningRate 0.9 0.999 1e-8 initialParameters :: IO (AdamState IO (Blob _) (Blob _))
    let go trainState = do
          grad <- objective (extractAdam trainState)
          trainState' <- updateAdam grad trainState
          next (extractAdam trainState') (go trainState')

    -- trainState <- initAMS (Blob.adamBlob { adam_α = learningRate }) (\a b -> pure $ Blob.zipBlob max a b) initialParameters
    -- let go trainState = do
    --       grad <- objective (extractAMS trainState)
    --       trainState' <- updateAMS grad trainState
    --       next (extractAMS trainState') (go trainState')

    go trainState


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

  case cmd of
    Train Nothing input savefile logfile chunkSize lr modelSize -> do
      withNat modelSize $ \(proxy :: Proxy modelSize) -> do
        initial_par <- sampleIO (Blob.generate $ uniform (-0.5) (0.5))
        deepseqM initial_par
        train proxy initial_par input savefile logfile chunkSize lr

    Train (Just resumepath) input savefile logfile chunkSize lr _ -> do
      (modelSize, box) <- LB.decode <$> LB.readFile resumepath
      withNat modelSize $ \(proxy :: Proxy modelSize) ->
        case openParBox box of
          Just initial_par -> do
            deepseqM initial_par
            train proxy initial_par input savefile logfile chunkSize lr
          Nothing -> error "model mismatch"

    Sample initpath length -> do
      (modelSize, box) <- LB.decode <$> LB.readFile initpath
      let n = fromMaybe 500 length
      withNat modelSize $ \(proxy :: Proxy modelSize) ->
        case openParBox box of
          Just initial -> do
            deepseqM initial
            let (par, c0) = Blob.split initial
                runrnn s c = do
                  ((c', o), _) <- Diff.runDiff (step proxy) (par, (s, c))
                  return (c', o)
            msg <- sampleRNN n (Blob.split c0) runrnn
            putStrLn msg
          Nothing -> error "model mismatch"
