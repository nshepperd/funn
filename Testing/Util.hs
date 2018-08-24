{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module Testing.Util where

import           Control.Category ((>>>))
import           Control.Monad
import           Control.Monad.Trans
import           Data.Functor.Identity
import           GHC.TypeLits
import           Test.QuickCheck hiding (scale)
import           Test.QuickCheck.Monadic

import qualified AI.Funn.CL.Blob as CLBlob
import           AI.Funn.CL.MonadCL (initOpenCL)
import qualified AI.Funn.CL.Network as Network
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Flat.Blob (Blob)
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space


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

magnitude x = inner x x

checkSame :: (Monad m,
              Finite m Double a, D a ~ a,
              Finite m Double b, D b ~ b,
              Arbitrary a, Arbitrary b,
              Show a, Show b, Eq b)
          => (m Property -> Property)
          -> Diff m a b
          -> Diff m a b
          -> Property
checkSame eval diff1 diff2 = monadic eval $ do
  a <- pick arbitrary
  asize <- lift (magnitude a)
  pre (asize < 1000)
  b1 <- lift (Diff.runDiffForward diff1 a)
  b2 <- lift (Diff.runDiffForward diff2 a)
  monitor (counterexample $ show (b1, b2))
  d <- lift (magnitude =<< minus b1 b2)
  m <- lift (magnitude =<< plus b1 b2)
  assert (d * 100000 <= m)


putR :: Applicative m => b -> Diff m a (a, b)
putR b = Diff run
  where
    run a = pure ((a, b), backward)
    backward (da, db) = pure da


-- Setup for OpenCL test cases.

clProperty :: IO Property -> Property
clProperty clProp = ioProperty (initOpenCL >> clProp)

checkGradientCL' :: (KnownNat n, KnownNat m) => Diff IO (Blob n) (Blob m) -> Property
checkGradientCL' diff = checkGradient 1e8 0.001 1e-4 clProperty diff

checkGradientCL :: (Loadable a n1, D a ~ a,
                    Loadable b n2, D b ~ b)
               => Diff IO a b -> Property
checkGradientCL diff = checkGradientCL' (fromCPUDiff >>> diff >>> fromGPUDiff)

checkSameCL :: (Loadable a n1, D a ~ a,
                Loadable b n2, D b ~ b,
                Loadable c n1, D c ~ c,
                Loadable d n2, D d ~ d)
            => Diff IO a b -> Diff IO c d -> Property
checkSameCL diff1 diff2 = (checkSame clProperty
                           (fromCPUDiff >>> diff1 >>> fromGPUDiff)
                           (fromCPUDiff >>> diff2 >>> fromGPUDiff))

checkGradientNet :: (Loadable a n1, D a ~ a,
                     Loadable b n2, D b ~ b,
                     KnownNat p)
                 => Network.Network IO p a b -> Property
checkGradientNet net = checkGradientCL (Network.toDiff net)

fromCPUDiff :: (Loadable a n, Loadable (D a) n) => Diff (IO) (Blob n) a
fromCPUDiff = Diff run
  where
    run a = do x <- fromCPU a
               return (x, backward)
    backward b = fromGPU b

fromGPUDiff :: (Loadable a n, Loadable (D a) n) => Diff (IO) a (Blob n)
fromGPUDiff = Diff run
  where
    run a = do x <- fromGPU a
               return (x, backward)
    backward b = fromCPU b

class KnownNat n => Loadable x n | x -> n where
  fromCPU :: Blob n -> IO x
  fromGPU :: x -> IO (Blob n)

instance Loadable Double 1 where
  fromCPU b = pure (head (Blob.toList b))
  fromGPU x = pure (Blob.fromList [x])

instance (KnownNat n, Floats a) => Loadable (CLBlob.Blob a n) n where
  fromCPU a = CLBlob.fromList (Blob.toList a)
  fromGPU a = Blob.fromList <$> CLBlob.toList a

instance (KnownDims ds, n ~ Prod ds) => Loadable (Tensor ds) n where
  fromCPU a = Tensor.fromList (Blob.toList a)
  fromGPU t = Blob.fromList <$> Tensor.toList t

instance (KnownNat n) => Loadable (Blob n) n where
  fromCPU a = return a
  fromGPU a = return a

instance (KnownNat m, Loadable a n1, Loadable b n2, m ~ (n1 + n2)) => Loadable (a, b) m where
  fromCPU ab = (,) <$> fromCPU a <*> fromCPU b
    where
      (a, b) = Blob.split ab
  fromGPU (a, b) = Blob.cat <$> fromGPU a <*> fromGPU b
