{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
import Prelude hiding (id)
import           Control.Category ((>>>), id)
import           Control.Monad.Trans
import           Data.Functor.Identity
import           Data.Proxy
import           GHC.TypeLits
import           Test.QuickCheck hiding (scale)
import           Test.QuickCheck.Gen
import           Test.QuickCheck.Gen.Unsafe
import           Test.QuickCheck.Monadic
import           Unsafe.Coerce

import           AI.Funn.Diff.Diff (Diff(..), Derivable(..), runDiffForward)
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Blob (Blob)
import qualified AI.Funn.CL.Blob as Blob
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.Flat
import           AI.Funn.CL.LSTM
import qualified AI.Funn.Flat.Blob as C
import           AI.Funn.Space

import Testing.Util

-- Setup for OpenCL test cases.

clProperty :: OpenCL Global Property -> Property
clProperty clProp = ioProperty (runOpenCLGlobal clProp)

-- Looser bounds for OpenCL as we are single precision.
checkGradientCL :: (Inner (OpenCL Global) Double a, D a ~ a,
                    Inner (OpenCL Global) Double b, D b ~ b,
                    Arbitrary a, Arbitrary b,
                    Show a, Show b,
                    Loadable a a',
                    Loadable b b'
                   )
               => Diff (OpenCL Global) a' b' -> Property
checkGradientCL diff = checkGradient 1e5 0.05 0.0008 clProperty (fromCPU >>> diff >>> fromGPU)

class Loadable x y | x -> y, y -> x where
  fromCPU :: Diff (OpenCL Global) x y
  fromGPU :: Diff (OpenCL Global) y x

instance Loadable Double Double where
  fromCPU = id
  fromGPU = id

instance (KnownNat n) => Loadable (C.Blob n) (Blob Global n) where
  fromCPU = Diff run
    where
      run a = do b <- Blob.fromList (C.toList a)
                 return (b, backward)
      backward db = C.fromList <$> Blob.toList db

  fromGPU = Diff run
    where
      run a = do b <- C.fromList <$> Blob.toList a
                 return (b, backward)
      backward db = Blob.fromList (C.toList db)

instance (Loadable a a1, Loadable b b1) => Loadable (a, b) (a1, b1) where
  fromCPU = Diff.first fromCPU >>> Diff.second fromCPU
  fromGPU = Diff.first fromGPU >>> Diff.second fromGPU

-- Buffer properties.

prop_Buffer_fromList :: Property
prop_Buffer_fromList = monadic clProperty $ do
  xs <- pick (arbitrary :: Gen [Double])
  buf <- lift $ Buffer.fromList xs
  ys <- lift $ Buffer.toList buf
  assert (xs == ys)

prop_Buffer_concat :: Property
prop_Buffer_concat = monadic clProperty $ do
  xs <- pick (arbitrary :: Gen [Double])
  ys <- pick (arbitrary :: Gen [Double])
  zs <- pick (arbitrary :: Gen [Double])
  buf1 <- lift $ Buffer.fromList xs
  buf2 <- lift $ Buffer.fromList ys
  buf3 <- lift $ Buffer.fromList zs
  buf <- lift $ Buffer.toList =<< Buffer.concat [buf1, buf2, buf3]
  assert (buf == xs ++ ys ++ zs)

prop_Buffer_slice :: Property
prop_Buffer_slice = monadic clProperty $ do
  xs <- pick (arbitrary :: Gen [Double])
  offset <- pick (choose (0, length xs))
  len <- pick (choose (0, length xs - offset))
  sub <- lift $ Buffer.slice offset len <$> Buffer.fromList xs
  ys <- lift $ Buffer.toList sub
  assert (ys == take len (drop offset xs))

-- Blob properties.

pickBlob :: (KnownNat n) => PropertyM (OpenCL Global) (Blob Global n)
pickBlob = lift . runDiffForward fromCPU =<< pick arbitrary

assertEqual :: (KnownNat n) => Blob Global n -> Blob Global n -> PropertyM (OpenCL Global) ()
assertEqual one two = do one_c <- lift (runDiffForward fromGPU one)
                         two_c <- lift (runDiffForward fromGPU two)
                         stop (one_c === two_c)

prop_Blob_plus_zero :: Property
prop_Blob_plus_zero = monadic clProperty $ do
  xs <- pickBlob @10
  z <- lift zero
  ys <- lift (plus xs z)
  assertEqual xs ys

prop_Blob_plus_comm :: Property
prop_Blob_plus_comm = monadic clProperty $ do
  xs <- pickBlob @10
  ys <- pickBlob @10
  z1 <- lift (plus xs ys)
  z2 <- lift (plus ys xs)
  assertEqual z1 z2

prop_Blob_plus :: Property
prop_Blob_plus = monadic clProperty $ do
  xs <- pickBlob @10
  ys <- pickBlob @10
  z1 <- lift (plus xs ys)
  xs_c <- lift (runDiffForward fromGPU xs)
  ys_c <- lift (runDiffForward fromGPU ys)
  z2 <- lift (runDiffForward fromCPU =<< plus xs_c ys_c)
  assertEqual z1 z2

prop_Blob_plusm :: Property
prop_Blob_plusm = monadic clProperty $ do
  xs <- pickBlob @10
  ys <- pickBlob @10
  z1 <- lift (plus xs ys)
  z2 <- lift (plusm [xs, ys])
  assertEqual z1 z2

prop_Blob_split :: Property
prop_Blob_split = monadic clProperty $ do
  zs <- pickBlob @9
  let (x1, y1) = Blob.splitBlob @2 @7 zs
  z1 <- lift (Blob.catBlob x1 y1)
  assertEqual zs z1


-- OpenCL flat diff

prop_fcdiff :: Property
prop_fcdiff = checkGradientCL (fcDiff @2 @2)

prop_reludiff :: Property
prop_reludiff = checkGradientCL (reluDiff @5)

prop_sigmoiddiff :: Property
prop_sigmoiddiff = checkGradientCL (sigmoidDiff @2)

prop_quadraticcost :: Property
prop_quadraticcost = checkGradientCL (quadraticCost @5)

prop_lstmdiff :: Property
prop_lstmdiff = checkGradientCL (lstmDiff @2)



-- Make TemplateHaskell aware of above definitions.
$(return [])

main :: IO Bool
main = $(quickCheckAll)
