{-# LANGUAGE BangPatterns #-}
module AI.Funn.SGD (sgd, defaultSave, vectorSource, sgd', sgd'') where

import           Control.Monad
import           Data.Foldable

import           Data.Coerce
import           Data.Functor.Identity
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import           Data.Random

import           Data.IORef

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe

import qualified Numeric.LinearAlgebra.HMatrix as HM

import           AI.Funn.Network

foreign import ccall "vector_add" ffi_vector_add :: CInt -> Ptr Double -> Ptr Double -> IO ()

sampleIO :: RVar a -> IO a
sampleIO v = runRVar v StdRandom

{-# NOINLINE vector_add #-}
vector_add :: M.IOVector Double -> S.Vector Double -> IO ()
vector_add tgt src = do M.unsafeWith tgt $ \tbuf -> do
                          S.unsafeWith src $ \sbuf -> do
                            ffi_vector_add (fromIntegral n) tbuf sbuf
  where
    n = V.length src

addToIO :: M.IOVector Double -> [Parameters] -> IO ()
addToIO target ys = go target (coerce ys :: [S.Vector Double])
  where
    go target [] = return ()
    go target (v:vs) = do
      vector_add target v
      go (M.drop (V.length v) target) vs

addTo :: Parameters -> [Parameters] -> Parameters
addTo (Parameters xs) ys = Parameters $ unsafePerformIO body
  where
    body = do target <- V.thaw xs
              addToIO target ys
              V.unsafeFreeze target

-- addParameters :: Parameters -> Parameters -> Parameters
-- addParameters (Parameters x) (Parameters y) = Parameters (x + y)

scaleParameters :: Double -> Parameters -> Parameters
scaleParameters x (Parameters y) = Parameters (HM.scale x y)

norm :: Parameters -> Double
norm (Parameters xs) = sqrt $ V.sum $ V.map (^2) xs

sgd :: (Monad m) => Double -> Network m p () -> Parameters -> m p -> (Int -> Parameters -> Double -> Double -> m ()) -> m ()
sgd lr network initial_pars source save = go initial_pars 0
  where
    go !pars !i = do
      input <- source

      (_, cost, k) <- evaluate network pars input
      (_, dpars) <- k ()
      let
        gpn = abs $ norm (fold dpars) / norm pars
      save i pars cost gpn
      let
        new_pars = pars `addTo` (map (scaleParameters (-lr)) dpars)
      go new_pars (i+1)

type LearningRate = Double
type Cost = Double

sgd' :: LearningRate -> Parameters -> Network Identity p () -> IO p -> IO [(Cost, Parameters)]
sgd' lr initial_pars network source = go initial_pars
  where
    go !pars = unsafeInterleaveIO $ do
      input <- source
      let
        (_, cost, k) = runIdentity $ evaluate network pars input
        (_, dpars) = runIdentity $ k ()
        new_pars = pars `addTo` (map (scaleParameters (-lr)) dpars)
      ((cost,pars) :) <$> go new_pars

sgd'' :: LearningRate -> Network Identity p () -> IO p -> IO [(Cost, Parameters)]
sgd'' lr network source = do initial_pars <- sampleIO (initialise network)
                             sgd' lr initial_pars network source

defaultSave :: IO (Int -> Parameters -> Double -> Double -> IO ())
defaultSave = go <$> newIORef 0 <*> newIORef 0
  where
    go ref_total ref_count i p c r = do
      modifyIORef' ref_total (smooth c)
      modifyIORef' ref_count (smooth 1)
      average_cost <- do total <- readIORef ref_total
                         count <- readIORef ref_count
                         return (total / count)
      when (i `mod` 100 == 0) $ do
        putStrLn $ show i ++ " " ++ show average_cost

    smooth x o = α * o + (1 - α) * x

    α = 0.99

vectorSource :: V.Vector v a => v a -> IO a
vectorSource values = do let n = V.length values
                         i <- runRVar (uniform 0 (n-1)) StdRandom
                         return (values V.! i)
