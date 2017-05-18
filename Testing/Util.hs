{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
module Testing.Util where

import Control.Category ((>>>))
import Control.Monad
import Control.Monad.Trans
import Data.Functor.Identity
import GHC.TypeLits
import Test.QuickCheck hiding (scale)
import Test.QuickCheck.Monadic

import AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import AI.Funn.Flat.Blob (Blob)
import qualified AI.Funn.Flat.Blob as Blob
import AI.Funn.Space

instance KnownNat n => Arbitrary (Blob n) where
  arbitrary = Blob.generate arbitrary

data DiffProp = DiffProp {
  cost :: Double,
  delta_finite :: Double,
  delta_backprop :: Double,
  diff_delta :: Double,
  mean_delta :: Double
  }
  deriving Show

linear :: (Monad m, Inner m Double b, D b ~ b)
       => b -> Diff m b Double
linear xb = Diff run
  where
    run b = do c <- inner b xb
               return (c, backward)
    backward δ = scale δ xb

measure :: (Show a, Monad m, Inner m Double a, D a ~ a)
        => Diff m a Double -> a -> a -> m DiffProp
measure diff a δa =
  do (l, k) <- runDiff diff a
     dL_da <- k 1
     δL_backprop <- δa `inner` dL_da
     l' <- Diff.runDiffForward diff =<< (a `plus` δa)
     let δL_finite = l' - l
     return (DiffProp l δL_finite δL_backprop
             (abs (δL_backprop - δL_finite) / 2)
             ((abs δL_backprop + abs δL_finite) / 2)
             )

rescale :: (Monad m, Inner m Double a) => Double -> a -> m a
rescale ε a = do
  cut_magnitude <- sqrt . max 1 <$> inner a a
  scale (ε / cut_magnitude) a

checkGradient :: (Monad m,
                  Inner m Double a, D a ~ a,
                  Inner m Double b, D b ~ b,
                  Arbitrary a, Arbitrary b,
                  Show a, Show b)
              => Double -> Double -> Double
              -> (m Property -> Property) -> Diff m a b -> Property
checkGradient min_precision wiggle ε eval diff = monadic eval $ do
  -- Get random input and perturbation.
  a <- pick arbitrary
  δa <- lift . rescale ε =<< pick arbitrary

  -- Precondition: avoid loss of precision in a + δa.
  a_mag <- lift (sqrt <$> inner a a)
  δa_mag <- lift (sqrt <$> inner δa δa)
  pre (δa_mag * min_precision > a_mag)

  -- Reduce to (a -> Double)
  xb <- pick arbitrary
  let new_diff = diff >>> linear xb

  -- Run gradient measurements
  stats <- lift (measure new_diff a δa)
  monitor (counterexample $ show stats)

  -- Precondition: exclude very small δL to avoid loss of precision in l - l'.
  pre (abs (delta_finite stats) * min_precision > abs (cost stats))
  -- Assert: finite gradient ~ backprop gradient.
  assert (diff_delta stats <= mean_delta stats * wiggle)

checkGradientI :: (Inner Identity Double a, D a ~ a,
                   Inner Identity Double b, D b ~ b,
                   Arbitrary a, Arbitrary b,
                   Show a, Show b)
               => Diff Identity a b -> Property
checkGradientI = checkGradient 1e8 0.01 0.00001 runIdentity

putR :: Applicative m => b -> Diff m a (a, b)
putR b = Diff run
  where
    run a = pure ((a, b), backward)
    backward (da, db) = pure da
