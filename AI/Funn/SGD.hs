{-# LANGUAGE BangPatterns, RecordWildCards #-}
module AI.Funn.SGD (sgd, SGDConfig(..), ssvrg, SSVRGConfig(..), adam, AdamConfig(..)) where

import           Control.Monad
import           Data.Foldable

import qualified Control.Monad.State.Lazy as SL
import           Data.Random
import           System.Random

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

data AdamConfig m p d = Adam {
  adam_α :: LearningRate,
  adam_β1 :: Momentum,
  adam_β2 :: Momentum,
  adam_ε :: Double,
  adam_pure_d :: Double -> m d,
  adam_scale_d :: Double -> d -> m d,
  adam_add_d :: d -> d -> m d,
  adam_square_d :: d -> m d,
  adam_sqrt_d :: d -> m d,
  adam_divide_d :: d -> d -> m d,
  adam_update_p :: p -> d -> m p
  }

adam :: (Monad m) => AdamConfig m p d -> p -> (p -> m d) -> (p -> m r -> m r) -> m r
adam Adam{..} p0 objective k = do
  m0 <- adam_pure_d 0
  v0 <- adam_pure_d 0
  ε <- adam_pure_d adam_ε
  let
    go p0 m0 v0 t0 = do
      let t = t0 + 1
      g <- objective p0
      m <- join $ adam_add_d <$> adam_scale_d adam_β1 m0 <*> adam_scale_d (1 - adam_β1) g
      g2 <- adam_square_d g
      v <- join $ adam_add_d <$> adam_scale_d adam_β2 v0 <*> adam_scale_d (1 - adam_β2) g2
      v2 <- adam_sqrt_d v
      let αt = adam_α * sqrt (1 - adam_β2^t) / (1 - adam_β1^t)
      update <- join (adam_divide_d <$> adam_scale_d (-αt) m <*> adam_add_d v2 ε)
      p <- adam_update_p p0 update
      k p (go p m v t)
  go p0 m0 v0 0
