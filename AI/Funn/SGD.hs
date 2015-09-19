{-# LANGUAGE BangPatterns, RecordWildCards #-}
module AI.Funn.SGD (sgd, sgd', SGDConfig(..), ssvrg, SSVRGConfig(..)) where

import           Control.Monad
import           Data.Foldable

import           Data.Coerce
import           Data.Functor.Identity
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import qualified Control.Monad.State.Lazy as SL
import           Data.Random
import           System.Random

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

addParameters :: Parameters -> Parameters -> Parameters
addParameters (Parameters x) (Parameters y) = Parameters (x + y)

scaleParameters :: Double -> Parameters -> Parameters
scaleParameters x (Parameters y) = Parameters (HM.scale x y)

type LearningRate = Double
type Momentum = Double
type Cost = Double

data SGDConfig m p i = SGDConfig {
  sgd_lr :: LearningRate,
  sgd_momentum :: Momentum,
  sgd_scale :: Double -> p -> p,
  sgd_add :: p -> p -> p,
  sgd_run :: p -> i -> m (Cost, p)
  }

sgd :: (Monad m) => SGDConfig m p i -> p -> [i] -> m [(Cost, p, p)]
sgd (SGDConfig{..}) initial_pars source = go initial_pars source Nothing
  where
    go pars [] _ = return []
    go pars (input:xs) moment = do
      (cost, dpars) <- sgd_run pars input
      let new_moment = case moment of
            Nothing -> sgd_scale (-sgd_lr) dpars
            Just p  -> sgd_scale (-sgd_lr) dpars `sgd_add` sgd_scale sgd_momentum p
          new_pars = pars `sgd_add` new_moment
      ((cost,pars,dpars) :) <$> go new_pars xs (Just new_moment)

sgd' :: (Monad m) => LearningRate -> Momentum -> Parameters -> Network m i () -> [i] -> m [(Cost, Parameters, Parameters)]
sgd' lr momentum initial_pars network source = sgd (SGDConfig lr momentum scaleParameters addParameters run) initial_pars source
  where
    run pars input = do
      (_, cost, k) <- evaluate network pars input
      (_, dpars) <- k ()
      return (cost, fold dpars)

data SSVRGConfig m p d i = SSVRGConfig {
  ssvrg_lr :: LearningRate,
  ssvrg_ks :: [Int],
  ssvrg_update_rate :: Int,
  ssvrg_sum :: [d] -> d,
  ssvrg_scale :: Double -> d -> d,
  ssvrg_add :: p -> d -> p,
  ssvrg_run :: p -> i -> m (Cost, d)
  }


{-# INLINE ssvrg #-}
ssvrg :: (Monad m) => SSVRGConfig m p d i -> p -> [i] -> m [(Cost, p, d)]
ssvrg (SSVRGConfig{..}) initial initial_source = part1 initial initial_source (zip ssvrg_ks updates)
  where
    part1 p _ [] = return []
    part1 p source ((k,m):ks) = do
      let (is, rest) = splitAt k source
      ds <- traverse (ssvrg_run p) is
      let ε = ssvrg_scale (1 / fromIntegral k) (ssvrg_sum (map snd ds))
      part2 p rest ks ε p m

    part2 w    is  ks ε p 0 = part1 p is ks
    part2 w (i:is) ks ε p m = do
      (cost, dp) <- ssvrg_run p i
      (_   , dw) <- ssvrg_run w i
      let d = ssvrg_sum [dp, ssvrg_scale (-1) dw, ε]
          new_p = p `ssvrg_add` ssvrg_scale (-ssvrg_lr) d
      ((cost, p, d) :) <$> (part2 w is ks ε new_p (m-1))

    updates = SL.evalState (let go = (:) <$> runRVar (uniform 1 ssvrg_update_rate) StdRandom <*> go in go) (mkStdGen 7)
