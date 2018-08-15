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
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

import           Control.Category ((>>>), id)
import           Control.Monad.Trans
import           Data.Functor.Identity
import           Data.Proxy
import qualified Data.Vector.Generic as V
import           GHC.TypeLits
import           Prelude hiding (id)
import           Test.QuickCheck hiding (scale)
import           Test.QuickCheck.Gen
import           Test.QuickCheck.Gen.Unsafe
import           Test.QuickCheck.Monadic
import           Unsafe.Coerce

import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.Flat
import           AI.Funn.CL.LSTM
import           AI.Funn.CL.Layers.Convolution
import           AI.Funn.CL.Layers.FullyConnected
import           AI.Funn.CL.Layers.Misc
import qualified AI.Funn.CL.Layers.Tensor as LT
import           AI.Funn.CL.Layers.Upscale
import           AI.Funn.CL.Mixing
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Network
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..), runDiffForward)
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as C
import qualified AI.Funn.Flat.Flat as C
import qualified AI.Funn.Flat.LSTM as C
import qualified AI.Funn.Flat.Mixing as C
import           AI.Funn.Space

import           Testing.Util

-- Setup for OpenCL test cases.

clProperty :: IO Property -> Property
clProperty clProp = ioProperty (initOpenCL >> clProp)

checkGradientCL :: (Loadable a n1, D a ~ a,
                    Loadable b n2, D b ~ b)
               => Diff IO a b -> Property
checkGradientCL diff = checkGradient 1e8 0.001 1e-4 clProperty (fromCPUDiff >>> diff >>> fromGPUDiff)

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
                 => Network IO p a b -> Property
checkGradientNet net = checkGradientCL (toDiff net)

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

instance (KnownDims ds, n ~ Prod ds) => Loadable (Tensor ds) n where
  fromCPU a = Tensor.fromList (C.toList a)
  fromGPU t = C.fromList <$> Tensor.toList t

instance (KnownNat n) => Loadable (C.Blob n) n where
  fromCPU a = return a
  fromGPU a = return a

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
  buf <- lift $ Buffer.toList (Buffer.concat [buf1, buf2, buf3])
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
  let z1 = Blob.catBlob x1 y1
  assertEqual zs z1

-- Tensor properties.

pickTensor :: forall ds. (KnownDims ds) => PropertyM IO (Tensor ds)
pickTensor = do blob <- pickBlob :: PropertyM IO (Blob Double (Prod ds))
                xs <- lift (Blob.toVector blob)
                lift (Tensor.fromVector xs)


tensorsEqual :: (KnownDims ds) => Tensor ds -> Tensor ds -> PropertyM IO ()
tensorsEqual one two = do one_c <- lift (Tensor.toVector one)
                          two_c <- lift (Tensor.toVector two)
                          stop (one_c === two_c)

prop_Tensor_sub_zero :: Property
prop_Tensor_sub_zero = monadic clProperty $ do
  xs <- pickTensor @[3,4]
  zs <- zero
  ys <- Tensor.subTensor xs zs
  tensorsEqual xs ys

prop_Tensor_plus_zero :: Property
prop_Tensor_plus_zero = monadic clProperty $ do
  xs <- pickTensor @[3,4]
  z <- zero
  ys <- plus xs z
  tensorsEqual xs ys

prop_Tensor_plus :: Property
prop_Tensor_plus = monadic clProperty $ do
  xs <- pickTensor @[3,4]
  ys <- pickTensor @[3,4]
  z1 <- plus xs ys
  xs_c <- Tensor.toVector xs
  ys_c <- Tensor.toVector ys
  let zs_c = V.zipWith (+) xs_c ys_c
  z2 <- Tensor.fromVector zs_c
  tensorsEqual z1 z2

prop_Tensor_mul :: Property
prop_Tensor_mul = monadic clProperty $ do
  xs <- pickTensor @[3,4]
  ys <- pickTensor @[3,4]
  z1 <- Tensor.mulTensor xs ys
  xs_c <- Tensor.toVector xs
  ys_c <- Tensor.toVector ys
  let zs_c = V.zipWith (*) xs_c ys_c
  z2 <- Tensor.fromVector zs_c
  tensorsEqual z1 z2

prop_Tensor_plusm :: Property
prop_Tensor_plusm = monadic clProperty $ do
  xs <- pickTensor @[3,4]
  ys <- pickTensor @[3,4]
  z1 <- plus xs ys
  z2 <- plusm [xs, ys]
  tensorsEqual z1 z2


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

prop_softmaxcost :: Property
prop_softmaxcost = checkGradientCL (putR 0 >>> softmaxCost @3 @IO @Double)

-- Tensor gradient

prop_iconv2d :: Property
prop_iconv2d = checkGradientCL (iconv2dDiff @2 @3 @3 @2 @2 @IO)

prop_conv2d :: Property
prop_conv2d = checkGradientCL (conv2dDiff @3 @1 @3 @3 @2 @2 @IO Proxy)

prop_doubleDiff :: Property
prop_doubleDiff = checkGradientCL (doubleDiff @2 @2 @2)


-- Tensor Net gradient

prop_conv2d_net :: Property
prop_conv2d_net = checkGradientNet (conv2d @3 @1 @3 @3 @2 @2 Proxy)

prop_fc_net :: Property
prop_fc_net = checkGradientNet (fcNet @3 @3)

prop_quadcost_net :: Property
prop_quadcost_net = checkGradientNet (LT.quadCostNet @[3,4])

prop_prelu_net :: Property
prop_prelu_net = checkGradientNet (LT.preluNet @'[3])

-- Equality

prop_fcdiff_eq :: Property
prop_fcdiff_eq = checkSameCL (fcDiff @2 @2 @Double) (C.fcDiff @2 @2)

prop_relu_eq :: Property
prop_relu_eq = checkSameCL (reluDiff @3 @IO @Double) (C.reluDiff)

prop_sigmoid_eq :: Property
prop_sigmoid_eq = checkSameCL (sigmoidDiff @3 @IO @Double) (C.sigmoidDiff)

prop_tanh_eq :: Property
prop_tanh_eq = checkSameCL (tanhDiff @3 @IO @Double) (C.tanhDiff)

prop_lstm_eq :: Property
prop_lstm_eq = checkSameCL (lstmDiff @3 @IO @Double) (C.lstmDiff @3)

prop_mix_eq :: Property
prop_mix_eq = checkSameCL (amixDiff @3 @2 @2 @Double Proxy) (C.amixDiff @3 @2 @2 Proxy)

prop_softmaxcost_eq :: Property
prop_softmaxcost_eq = checkSameCL (putR 0 >>> softmaxCost @3 @IO @Double) (putR 0 >>> C.softmaxCost)


-- Make TemplateHaskell aware of above definitions.
$(return [])

main :: IO Bool
main = $(quickCheckAll)
