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
{-# LANGUAGE TypeOperators #-}
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
import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.Flat
import           AI.Funn.CL.LSTM
import           AI.Funn.CL.Mixing
import qualified AI.Funn.Flat.Blob as C
import           AI.Funn.Space

import Testing.Util

-- Setup for OpenCL test cases.

clProperty :: IO Property -> Property
clProperty clProp = ioProperty (initOpenCL >> clProp)

checkGradientCL :: (Loadable a n1, D a ~ a,
                    Loadable b n2, D b ~ b)
               => Diff IO a b -> Property
checkGradientCL diff = checkGradient 1e8 0.001 1e-4 clProperty (fromCPUDiff >>> diff >>> fromGPUDiff)

fromCPUDiff :: (Loadable a n, Loadable (D a) n) => Diff (IO) (C.Blob n) a
fromCPUDiff = Diff run
  where
    run a = do x <- fromCPU a
               return (x, backward)
    backward b = fromGPU b

fromGPUDiff :: (Loadable a n, Loadable (D a) n) => Diff (IO) a (C.Blob n)
fromGPUDiff = Diff run
  where
    run a = do x <- fromGPU a
               return (x, backward)
    backward b = fromCPU b

class KnownNat n => Loadable x n | x -> n where
  fromCPU :: C.Blob n -> IO x
  fromGPU :: x -> IO (C.Blob n)

instance Loadable Double 1 where
  fromCPU b = pure (head (C.toList b))
  fromGPU x = pure (C.fromList [x])

instance (KnownNat n, Floats a) => Loadable (Blob a n) n where
  fromCPU a = Blob.fromList (C.toList a)
  fromGPU a = C.fromList <$> Blob.toList a

instance (KnownNat m, Loadable a n1, Loadable b n2, m ~ (n1 + n2)) => Loadable (a, b) m where
  fromCPU ab = (,) <$> fromCPU a <*> fromCPU b
    where
      (a, b) = C.split ab
  fromGPU (a, b) = C.cat <$> fromGPU a <*> fromGPU b

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

pickBlob :: (KnownNat n) => PropertyM IO (Blob Double n)
pickBlob = lift . fromCPU =<< pick arbitrary

assertEqual :: (KnownNat n) => Blob Double n -> Blob Double n -> PropertyM IO ()
assertEqual one two = do one_c <- lift (fromGPU one)
                         two_c <- lift (fromGPU two)
                         stop (one_c === two_c)

prop_Blob_sub_zero :: Property
prop_Blob_sub_zero = monadic clProperty $ do
  xs <- pickBlob @10
  z <- lift zero
  ys <- lift (subBlob xs z)
  assertEqual xs ys

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
  xs_c <- lift (fromGPU xs)
  ys_c <- lift (fromGPU ys)
  z2 <- lift (fromCPU =<< plus xs_c ys_c)
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
prop_fcdiff = checkGradientCL (fcDiff @2 @2 @Double)

prop_reludiff :: Property
prop_reludiff = checkGradientCL (reluDiff @5 @IO @Double)

prop_sigmoiddiff :: Property
prop_sigmoiddiff = checkGradientCL (sigmoidDiff @3 @IO @Double)

prop_tanhdiff :: Property
prop_tanhdiff = checkGradientCL (tanhDiff @3 @IO @Double)

prop_quadraticcost :: Property
prop_quadraticcost = checkGradientCL (quadraticCost @5 @IO @Double)

prop_lstmdiff :: Property
prop_lstmdiff = checkGradientCL (lstmDiff @2 @IO @Double)

prop_mixdiff :: Property
prop_mixdiff = checkGradientCL (amixDiff @3 @2 @2 @Double Proxy)

-- Make TemplateHaskell aware of above definitions.
$(return [])

main :: IO Bool
main = $(quickCheckAll)
