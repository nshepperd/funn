{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards #-}
module AI.Funn.Optimizer.Adam (AdamState(adam_α, adam_β1, adam_β2),
                               initAdam, extractAdam, updateAdam,
                               AdamOps(..), Adam(..), defaultAdam) where

import Control.Monad
import Data.Foldable

import AI.Funn.Space

type LearningRate = Double
type Momentum = Double

class (VectorSpace m Double d) => AdamOps m d where
  adam_pure_d :: Double -> m d
  adam_square_d :: d -> m d
  adam_sqrt_d :: d -> m d
  adam_divide_d :: d -> d -> m d

class (AdamOps m d) => Adam m d p | p -> d where
  adam_update_p :: p -> d -> m p


data AdamState m d p = AdamState {
  adam_α :: LearningRate,
  adam_β1 :: Momentum,
  adam_β2 :: Momentum,
  ε :: d,
  x :: p,
  m :: d,
  v :: d,
  t :: Int
  }

initAdam :: (Adam m d p, Monad m) => LearningRate -> Momentum -> Momentum -> Double -> p -> m (AdamState m d p)
initAdam α β1 β2 ε x0 = do
  m0 <- zero
  v0 <- zero
  ε0 <- adam_pure_d ε
  return $ AdamState {
    adam_α = α,
    adam_β1 = β1,
    adam_β2 = β2,
    ε = ε0,
    x = x0,
    m = m0,
    v = v0,
    t = 1
    }

extractAdam :: AdamState m d p -> p
extractAdam state = x state

updateAdam :: (Adam m d p, Monad m) => d -> AdamState m d p -> m (AdamState m d p)
updateAdam d (AdamState{..}) = do
  let t' = t + 1
  m' <- join $ plus <$> scale adam_β1 m <*> scale (1 - adam_β1) d
  d2 <- adam_square_d d
  v' <- join $ plus <$> scale adam_β2 v <*> scale (1 - adam_β2) d2
  v2 <- adam_sqrt_d v'
  let αt = adam_α * sqrt (1 - adam_β2^t) / (1 - adam_β1^t)
  update <- join (adam_divide_d <$> scale (-αt) m' <*> plus v2 ε)
  x' <- adam_update_p x update
  return $ AdamState {
    x = x',
    m = m',
    v = v',
    t = t',
    ..
    }

defaultAdam :: (Adam m d p, Monad m) => p -> m (AdamState m d p)
defaultAdam = initAdam 0.001 0.9 0.999 1e-8
