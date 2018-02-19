{-# LANGUAGE BangPatterns, RecordWildCards, FlexibleContexts #-}
module AI.Funn.Optimizer.Adam (AdamState, initAdam, extractAdam, updateAdam) where

import Control.Monad
import Data.Foldable

import AI.Funn.SGD
import AI.Funn.Space

type LearningRate = Double
type Momentum = Double

data AdamState m d p = AdamState {
  adamConfig :: AdamConfig m p d,
  ε :: d,
  x :: p,
  m :: d,
  v :: d,
  t :: Int
  }

initAdam :: (Monad m) => AdamConfig m p d -> p -> m (AdamState m d p)
initAdam conf x0 = do
  m0 <- adam_pure_d conf 0
  v0 <- adam_pure_d conf 0
  ε0 <- adam_pure_d conf (adam_ε conf)
  return $ AdamState {
    adamConfig = conf,
    ε = ε0,
    x = x0,
    m = m0,
    v = v0,
    t = 1
    }

extractAdam :: AdamState m d p -> p
extractAdam = x

updateAdam :: (Monad m) => d -> AdamState m d p -> m (AdamState m d p)
updateAdam d (AdamState{adamConfig = adamConfig@(Adam{..}), ..}) = do
  let t' = t + 1
  m' <- join $ adam_add_d <$> adam_scale_d adam_β1 m <*> adam_scale_d (1 - adam_β1) d
  d2 <- adam_square_d d
  v' <- join $ adam_add_d <$> adam_scale_d adam_β2 v <*> adam_scale_d (1 - adam_β2) d2
  v2 <- adam_sqrt_d v'
  let αt = adam_α * sqrt (1 - adam_β2^t) / (1 - adam_β1^t)
  update <- join (adam_divide_d <$> adam_scale_d (-αt) m' <*> adam_add_d v2 ε)
  x' <- adam_update_p x update
  return $ AdamState {
    adamConfig = adamConfig,
    ε = ε,
    x = x',
    m = m',
    v = v',
    t = t'
    }
