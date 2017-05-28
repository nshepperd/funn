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

minus :: (Monad m, VectorSpace m Double a) => a -> a -> m a
minus a b = plus a =<< scale (-1) b

measure :: (Show a, Monad m, Inner m Double a, D a ~ a)
        => Diff m a Double -> a -> a -> m DiffProp
measure diff a δa =
  do (lc, k) <- runDiff diff a
     dL_da <- k 1
     ar <- plus a δa
     al <- minus a δa
     δa_true <- minus ar al
     δL_backprop <- δa_true `inner` dL_da
     l1 <- Diff.runDiffForward diff ar
     l2 <- Diff.runDiffForward diff al
     let δL_finite = l1 - l2
     return (DiffProp lc δL_finite δL_backprop
             (abs (δL_backprop - δL_finite) / 2)
             ((abs δL_backprop + abs δL_finite) / 2)
             )

rescale :: (Monad m, Inner m Double a) => Double -> a -> m a
rescale ε a = do
  cut_magnitude <- sqrt . max 1 <$> inner a a
  scale (ε / cut_magnitude) a

checkGradient :: (Monad m,
                  Finite m Double a, D a ~ a,
                  Finite m Double b, D b ~ b,
                  Arbitrary a, Arbitrary b,
                  Show a, Show b)
              => Double -> Double -> Double
              -> (m Property -> Property) -> Diff m a b -> Property
checkGradient min_precision wiggle ε eval diff = monadic eval $ do
  -- Get random input and perturbation.
  a <- pick arbitrary
  δa <- lift . rescale ε =<< pick arbitrary

  -- Precondition: avoid loss of precision in a + δa.
  a_parts <- lift (getBasis a)
  δa_parts <- lift (getBasis δa)
  pre (and $ zipWith (\a δa -> abs δa * min_precision > abs a) a_parts δa_parts)

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

checkGradientI :: (Finite Identity Double a, D a ~ a,
                   Finite Identity Double b, D b ~ b,
                   Arbitrary a, Arbitrary b,
                   Show a, Show b)
               => Diff Identity a b -> Property
checkGradientI = checkGradient 1e8 0.001 1e-4 runIdentity

putR :: Applicative m => b -> Diff m a (a, b)
putR b = Diff run
  where
    run a = pure ((a, b), backward)
    backward (da, db) = pure da
