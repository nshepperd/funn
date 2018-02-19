{-# LANGUAGE BangPatterns, RecordWildCards, FlexibleContexts #-}
module AI.Funn.Optimizer.SGD (SGDState, initSGD, extractSGD, updateSGD) where

import Control.Monad
import Data.Foldable

import AI.Funn.Space

type LearningRate = Double
type Momentum = Double

data SGDState m d p = SGDState {
  sgdStepSize :: LearningRate,
  sgdMomentumWeight :: Momentum,
  sgdScale :: Double -> d -> m d,
  sgdAddDP :: d -> p -> m p,
  sgdAddDD :: d -> d -> m d,
  sgdValue :: p,
  sgdMoment :: d
  }

initSGD :: (Monad m, VectorSpace m Double d) => LearningRate -> Momentum -> (d -> p -> m p) -> p -> m (SGDState m d p)
initSGD lr mr add x0 = do
  d0 <- zero
  return $ SGDState {
    sgdStepSize = lr,
    sgdMomentumWeight = mr,
    sgdScale = scale,
    sgdAddDP = add,
    sgdAddDD = plus,
    sgdValue = x0,
    sgdMoment = d0
    }

extractSGD :: SGDState m d p -> p
extractSGD = sgdValue

updateSGD :: (Monad m) => d -> SGDState m d p -> m (SGDState m d p)
updateSGD d (SGDState{..}) = do
  newMoment <- join $ sgdAddDD <$> sgdScale sgdMomentumWeight sgdMoment <*> sgdScale (-sgdStepSize) d
  newValue <- sgdAddDP newMoment sgdValue
  return $ SGDState {
    sgdStepSize = sgdStepSize,
    sgdMomentumWeight = sgdMomentumWeight,
    sgdScale = sgdScale,
    sgdAddDP = sgdAddDP,
    sgdAddDD = sgdAddDD,
    sgdValue = newValue,
    sgdMoment = newMoment
    }
