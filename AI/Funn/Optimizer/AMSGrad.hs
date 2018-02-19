{-# LANGUAGE BangPatterns, RecordWildCards, FlexibleContexts #-}
module AI.Funn.Optimizer.AMSGrad (AMSState, initAMS, extractAMS, updateAMS) where

import Control.Monad
import Data.Foldable

import AI.Funn.SGD
import AI.Funn.Space

type LearningRate = Double
type Momentum = Double

data AMSState m d p = AMSState {
  adamConfig :: AdamConfig m p d,
  max_d :: d -> d -> m d,
  ε :: d,
  x :: p,
  m :: d,
  v :: d,
  t :: Int
  }

initAMS :: (Monad m) => AdamConfig m p d -> (d -> d -> m d) -> p -> m (AMSState m d p)
initAMS conf max_d x0 = do
  m0 <- adam_pure_d conf 0
  v0 <- adam_pure_d conf 0
  ε0 <- adam_pure_d conf (adam_ε conf)
  return $ AMSState {
    adamConfig = conf,
    max_d = max_d,
    ε = ε0,
    x = x0,
    m = m0,
    v = v0,
    t = 1
    }

extractAMS :: AMSState m d p -> p
extractAMS = x

updateAMS :: (Monad m) => d -> AMSState m d p -> m (AMSState m d p)
updateAMS d (AMSState{adamConfig = adamConfig@(Adam{..}), ..}) = do
  let t' = t + 1
  m' <- join $ adam_add_d <$> adam_scale_d adam_β1 m <*> adam_scale_d (1 - adam_β1) d
  d2 <- adam_square_d d
  v'1 <- join $ adam_add_d <$> adam_scale_d adam_β2 v <*> adam_scale_d (1 - adam_β2) d2
  v' <- max_d v v'1
  v2 <- adam_sqrt_d v'
  let αt = adam_α * sqrt (1 - adam_β2^t) / (1 - adam_β1^t)
  update <- join (adam_divide_d <$> adam_scale_d (-αt) m' <*> adam_add_d v2 ε)
  x' <- adam_update_p x update
  return $ AMSState {
    adamConfig = adamConfig,
    max_d = max_d,
    ε = ε,
    x = x',
    m = m',
    v = v',
    t = t'
    }
