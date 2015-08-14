{-# LANGUAGE BangPatterns #-}
module AI.Funn.SGD (sgd, defaultSave) where

import           Control.Monad
import           Data.Foldable

import           Data.Coerce
import           Data.Functor.Identity
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import           Data.IORef

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe

import qualified Numeric.LinearAlgebra.HMatrix as HM

import           AI.Funn.Network

foreign import ccall "vector_add" ffi_vector_add :: CInt -> Ptr Double -> Ptr Double -> IO ()

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

defaultSave :: IO (Int -> Parameters -> Double -> Double -> IO ())
defaultSave = go <$> newIORef 0
  where
    go ref i p c r = do
      modifyIORef' ref (\x -> 0.99 * x + 0.01 * c)
      average_cost <- readIORef ref
      when (i `mod` 100 == 0) $ do
        putStrLn $ show i ++ " " ++ show average_cost
