{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, ForeignFunctionInterface #-}
{-# LANGUAGE TypeApplications, PartialTypeSignatures #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}

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
import           Debug.Trace
import           GHC.TypeLits

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe

import qualified Control.Monad.State.Lazy as SL
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
import           AI.Funn.Diff.Diff (Diff(..), Additive(..), Derivable(..), (>>>))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Diff.Pointed
import           AI.Funn.Diff.RNN
import           AI.Funn.Flat.Flat
import           AI.Funn.Flat.LSTM
import           AI.Funn.Flat.Mixing
import           AI.Funn.SGD

sampleIO :: RVar a -> IO a
sampleIO v = runRVar v StdRandom

deepseqM :: (Monad m, NFData a) => a -> m ()
deepseqM x = deepseq x (return ())

average :: [Double] -> Double
average xs = sum xs / genericLength xs
stdev :: [Double] -> Double
stdev xs = let m = average xs in
            sum [(x-m)^2 | x <- xs] / (genericLength xs - 1)

checkGradient' :: forall a. (KnownNat a) => Diff IO (Blob a) Double -> IO (Double, Double, Double)
checkGradient' network = do gs <- replicateM 1000 (checkGradient network)
                            let
                              ls = map log gs
                              x = average ls
                              d = stdev ls
                            return (exp (x - d), exp x, exp (x + d))

checkGradient :: forall a. (KnownNat a) => Diff IO (Blob a) Double -> IO Double
checkGradient network = do input <- sampleIO (generateBlob $ uniform 0 1)
                           (e, k) <- runDiff network input
                           d_input <- k 1
                           perturb <- sampleIO (generateBlob $ uniform (-ε) ε)
                           let input' = input Diff.## perturb
                           (e', _) <- runDiff network input'
                           let δ_gradient = V.sum (V.zipWith (*) (getBlob d_input) (getBlob perturb))
                               δ_finite = e' - e
                           print (input, e, d_input)
                           return $ abs (δ_gradient - δ_finite) / (abs δ_gradient + abs δ_finite)

  where
    a = fromIntegral (natVal (Proxy :: Proxy a)) :: Int
    ε = 0.00001

adamBlob :: forall m n. (Monad m, KnownNat n) => AdamConfig m (Blob n) (Blob n)
adamBlob = Adam {
  adam_α = 0.001,
  adam_β1 = 0.9,
  adam_β2 = 0.999,
  adam_ε = 1e-8,
  adam_pure_d = \x -> generateBlob (pure x),
  adam_scale_d = \x b -> pure (scaleBlob x b),
  adam_add_d = plus,
  adam_square_d = \(Blob b) -> pure $ Blob (V.map (^2) b),
  adam_sqrt_d = \(Blob b) -> pure $ Blob (V.map sqrt b),
  adam_divide_d = \(Blob x) (Blob y) -> pure $ Blob (V.zipWith (/) x y),
  adam_update_p = plus
  }
  where
    n = fromIntegral (natVal (Proxy :: Proxy n))

sampleRNN :: Int -> s -> (s -> Blob 128 -> IO (s, Blob 128)) -> IO [Char]
sampleRNN n s0 next = go n s0 (unsafeBlob (1 : replicate 127 0))
  where
    go 0 _ _ = return []
    go n s q = do
      (new_s, new_q) <- next s q
      let exps = V.map exp (getBlob new_q)
          factor = 1 / V.sum exps
          ps = V.map (*factor) exps
      c <- sampleIO (categorical $ zip (V.toList ps) [0 :: Int ..])
      rest <- go (n-1) new_s (oneofn V.! fromIntegral c)
      return (chr c : rest)

    oneofn :: Vector (Blob 128)
    oneofn = V.generate 128 (\i -> unsafeBlob (replicate i 0 ++ [1] ++ replicate (127 - i) 0))


data Commands = Train (Maybe FilePath) FilePath (Maybe FilePath) (Maybe FilePath) Int Double
              | Sample FilePath (Maybe Int)
              | CheckDeriv
              deriving (Show)

type H = 200

instance (Additive m a, Monad m) => Additive m (Vector a) where
  zero = pure V.empty
  plus xs ys
    | V.null xs = pure ys
    | V.null ys = pure xs
    | otherwise = V.zipWithM plus xs ys

instance LB.Binary (Blob n) where
  put (Blob xs) = putVector putDouble xs
  get = Blob <$> getVector getDouble

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
    step :: Diff IO (Blob _, ((Blob H, Blob H), Blob 128)) ((Blob H, Blob H), Blob 128)
    step = runPointed $ \in_all -> do
      (Var pars, ((Var hidden, Var prev), Var char)) <- unpack in_all
      (Var p1, Var pars1) <- splitDiff <-- pars
      Var combined_in <- mergeDiff <-- (prev, char)
      Var lstm_in <- (tanhDiff <<< amixDiff (Proxy @ 5)) <-- (p1, combined_in)
      (Var p2, Var p3) <- splitDiff <-- pars1
      (Var hidden', Var lstm_out) <- lstmDiff <-- (p2, (hidden, lstm_in))
      -- (Var p3, Var pars3) <- splitDiff <-- pars2
      Var final_dist <- amixDiff (Proxy @ 5) <-- (p3, lstm_out)
      pack ((hidden', lstm_out), final_dist)

    network :: Diff IO (Blob _, Vector (Blob 128)) (Vector (Blob 128))
    network = runPointed $ \in_all -> do
      (Var pars, Var inputs) <- unpack in_all
      (Var step_pars, Var init) <- splitDiff <-- pars
      (Var h0, Var c0) <- splitDiff <-- init
      (Var s, Var o) <- scanlDiff step <-- (step_pars, ((h0, c0), inputs))
      return o

    eval :: Diff IO ((Blob _, Vector (Blob 128)), Vector Int) Double
    eval = Diff.first network >>> zipDiff >>> mapDiff softmaxCost >>> vsumDiff

    oneofn :: Vector (Blob 128)
    oneofn = V.generate 128 (\i -> unsafeBlob (replicate i 0 ++ [1] ++ replicate (127 - i) 0))

    runrnn par s c = do
      ((c', o), _) <- Diff.runDiff step (par, (s, c))
      return (c', o)


  case cmd of
   Train initpath input savefile logfile chunkSize lr -> do
     initial_par <- case initpath of
       Just path -> LB.decode <$> LB.readFile path
       Nothing -> sampleIO (generateBlob $ uniform (-0.5) (0.5))

     deepseqM initial_par

     text <- B.readFile input

     running_average <- newIORef 0
     running_count <- newIORef 0
     iteration <- newIORef (0 :: Int)

     startTime <- getTime ProcessCPUTime

     let
       α :: Double
       α = 0.99

       tvec = V.fromList (filter (<128) $ B.unpack text) :: U.Vector Word8
       ovec = V.map (\c -> oneofn V.! (fromIntegral c)) (V.convert tvec) :: Vector (Blob 128)

       source :: IO (Vector (Blob 128), Vector Int)
       source = do s <- sampleIO (uniform 0 (V.length tvec - chunkSize))
                   let
                     input = (oneofn V.! 0) `V.cons` V.slice s (chunkSize - 1) ovec
                     output = V.map fromIntegral (V.convert (V.slice s chunkSize tvec))
                   return (input, output)

       objective p = do
         (i, o) <- source
         (err, k) <- Diff.runDiff eval ((p,i),o)

         when (not (isInfinite err || isNaN err)) $ do
           modifyIORef' running_average (\x -> (α*x + (1 - α)*err))
           modifyIORef' running_count (\x -> (α*x + (1 - α)*1))

         putStrLn $ "Error: " ++ show err
         ((dp, _), _) <- k 1
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
           let (par, c0) = splitBlob p
           msg <- sampleRNN 200 (splitBlob c0) (runrnn par)
           putStrLn msg

         when (i `mod` 1000 == 0) $ do
           case savefile of
            Just savefile -> do
              LB.writeFile (printf "%s-%6.6i-%5.5f.bin" savefile i x) $ LB.encode p
              LB.writeFile (savefile ++ "-latest.bin") $ LB.encode p
            Nothing -> return ()

         m

     deepseqM (tvec, ovec)

     adam (adamBlob { adam_α = lr }) initial_par objective next

   Sample initpath length -> do
     initial <- LB.decode <$> LB.readFile initpath
     let n = case length of
           Just n -> n
           Nothing -> 500

     let (par, c0) = splitBlob initial
     msg <- sampleRNN n (splitBlob c0) (runrnn par)
     putStrLn msg
