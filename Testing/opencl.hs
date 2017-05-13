{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
import           Control.Category ((>>>))
import           Control.Monad.Trans
import           Data.Functor.Identity
import           Data.Proxy
import           GHC.TypeLits
import           Test.QuickCheck hiding (scale)
import           Test.QuickCheck.Gen
import           Test.QuickCheck.Gen.Unsafe
import           Test.QuickCheck.Monadic
import           Unsafe.Coerce

import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Blob (Blob)
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.Flat
import qualified AI.Funn.Flat.Blob as C
import           AI.Funn.Space

import Testing.Util

-- Setup for OpenCL test cases.

clProperty :: (forall s. OpenCL s Property) -> Property
clProperty clProp = ioProperty (runOpenCL clProp)

data T

clPropertyUnsafe :: OpenCL T Property -> Property
clPropertyUnsafe f = clProperty (unsafeCoerce f)

-- Looser bounds for OpenCL as we are single precision.
checkGradientCL :: (Inner (OpenCL T) Double a, D a ~ a,
                    Inner (OpenCL T) Double b, D b ~ b,
                    Arbitrary a, Arbitrary b,
                    Show a, Show b)
               => (forall s. Diff (OpenCL s) a b) -> Property
checkGradientCL diff = checkGradient 1e5 0.05 0.001 clPropertyUnsafe diff

fromCPU :: (MonadCL s m, KnownNat n) => Diff m (C.Blob n) (Blob s n)
fromCPU = Diff run
  where
    run a = do b <- Blob.fromList (C.toList a)
               return (b, backward)
    backward db = C.fromList <$> Blob.toList db

toCPU :: (MonadCL s m, KnownNat n) => Diff m (Blob s n) (C.Blob n)
toCPU = Diff run
  where
    run a = do b <- C.fromList <$> Blob.toList a
               return (b, backward)
    backward db = Blob.fromList (C.toList db)

-- OpenCL flat blob stuff

prop_fcdiff :: Property
prop_fcdiff = checkGradientCL ((Diff.first fromCPU >>> Diff.second fromCPU) >>> (fcDiff @1 @1) >>> toCPU)
  -- where
  --   d :: Diff (OpenCL s) (Blob s _, Blob s 4) (Blob s 5)
  --   d = fcDiff
  --   -- ((fromCPU *** fromCPU) >>> _ (fcDiff) >>> toCPU)


-- Make TemplateHaskell aware of above definitions.
$(return [])

main :: IO Bool
main = $(quickCheckAll)
