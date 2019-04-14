module AI.Funn.Common where

import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S

isBad :: (RealFloat a) => a -> Bool
isBad x = isNaN x || isInfinite x

class CheckNAN a where
  check :: (Show b) => String -> a -> b -> ()

instance (S.Storable a, RealFloat a) => CheckNAN (S.Vector a) where
  {-# INLINE check #-}
  check s xs b = ()
  -- check s xs b = if V.any (\x -> isNaN x || isInfinite x) xs then
  --                  error ("[" ++ s ++ "] checkNaN -- " ++ show b)
  --                else ()

instance (CheckNAN a, CheckNAN b) => CheckNAN (a,b) where
  check s (x,y) b = check s x b `seq` check s y b


data RunningAverage = Avg Double Double Double deriving (Show)

newRunningAverage :: Double -> RunningAverage
newRunningAverage alpha = Avg alpha 0 0

updateRunningAverage :: Double -> RunningAverage -> RunningAverage
updateRunningAverage x (Avg alpha total count) = Avg alpha new_total new_count
  where
    new_count = (alpha * count + (1 - alpha) * 1)
    new_total = (alpha * total + (1 - alpha) * x)

readRunningAverage :: RunningAverage -> Double
readRunningAverage (Avg alpha total count)
  | count > 0 = total / count
  | otherwise = 0.0
