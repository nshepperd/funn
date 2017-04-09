{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, ForeignFunctionInterface #-}
{-# LANGUAGE TypeApplications, PartialTypeSignatures #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}

import           Control.Applicative
import           Control.Monad
import           Control.Category ((>>>))
import           Data.Foldable
import           Data.Monoid
import           Data.Traversable
import           Data.Functor.Identity

import           Data.Char
import           Data.IORef
import           Data.List
import           Data.Maybe
import           Data.Proxy
import           Data.Word

import           Control.Concurrent
import           Control.DeepSeq
import           Control.Monad.IO.Class
import qualified Control.Monad.State.Lazy as SL
import           Debug.Trace
import           Foreign.C
import qualified Foreign.OpenCL.Bindings as CL
import           Foreign.Ptr
import           GHC.TypeLits
import           System.Clock
import           System.Environment
import           System.IO
import           System.Mem

import           Data.Random
import           Data.Random.Distribution.Categorical
import           System.Random

import           Options.Applicative
import           Text.Printf

import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.Flat
import           AI.Funn.CL.MonadCL
import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Additive(..), Derivable(..), (>>>))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Diff.Pointed
import           AI.Funn.Diff.RNN
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

adamBlob :: KnownNat n => AdamConfig (OpenCL s) (Blob s n) (Blob s n)
adamBlob = defaultAdam {
  adam_pure_d = pureBlob,
  adam_scale_d = scaleBlob,
  adam_add_d = addBlob,
  adam_square_d = squareBlob,
  adam_sqrt_d = sqrtBlob,
  adam_divide_d = divideBlob,
  adam_update_p = addBlob
  }

data MovingAverage = MV Double (IORef Double) (IORef Double)

newMovingAverage :: Double -> IO MovingAverage
newMovingAverage α = MV α <$> newIORef 0 <*> newIORef 0

updateAverage :: Double -> MovingAverage -> IO ()
updateAverage v (MV α average count) = when (not (isInfinite v || isNaN v)) $ do
  modifyIORef' average (\x -> (α*x + (1 - α)*v))
  modifyIORef' count (\x -> (α*x + (1 - α)*1))

getAverage :: MovingAverage -> IO Double
getAverage (MV α average count) = do
  q <- readIORef average
  w <- readIORef count
  return (q / w)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  -- runOpenCL $ do
  --   pars <- Blob.fromList [1, 1, 1, 0, 10, 20];
  --   xs <- Blob.fromList [3, 4];
  --   (o, _) <- runDiff (fcDiff @2 @2) (pars, xs);
  --   Blob.toList o >>= liftIO . print

  running_average <- newMovingAverage 0.99
  iteration <- newIORef (0 :: Int)
  startTime <- getTime ProcessCPUTime

  let
    network :: Diff (OpenCL s) (Blob s 3, Blob s 3) Double
    network = Diff.first sigmoidDiff >>> quadraticCost

    objective :: Blob s 3 -> Blob s 3 -> OpenCL s (Blob s 3)
    objective o p = do
      (err, k) <- Diff.runDiff network (p, o)
      liftIO $ updateAverage err running_average
      liftIO $ putStrLn $ "Error: " ++ show err
      (dp, _) <- k 1
      return dp

    next :: Blob s 3 -> OpenCL s r -> OpenCL s r
    next p m = do
      values <- Blob.toList p
      liftIO $ do
        x <- getAverage running_average
        modifyIORef' iteration (+1)
        i <- readIORef iteration
        now <- getTime ProcessCPUTime
        let tdiff = fromIntegral (toNanoSecs (now - startTime)) / (10^9) :: Double
        putStrLn $ printf "[% 11.4f | %i]  %f %s" tdiff i x (show values)
        -- threadDelay (100000)
      m

  runOpenCL $ do
    dev <- getDeviceID
    liftIO $ print =<< CL.deviceVersion dev
    liftIO $ print =<< CL.deviceExtensions dev
    liftIO $ print =<< CL.deviceGlobalMemSize dev
    liftIO $ print =<< CL.deviceMaxMemAllocSize dev
    o <- Blob.fromList @3 [0.3, 0.9, 0.5]
    initial <- Blob.fromList @3 [1, 2, 3]
    adam adamBlob initial (objective o) next

  performMinorGC
