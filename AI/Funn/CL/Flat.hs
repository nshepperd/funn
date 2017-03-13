{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TemplateHaskell #-}
module AI.Funn.CL.Flat (
  reluDiff, sigmoidDiff,
  fcDiff, quadraticCost
  ) where

import           Control.Applicative
import           Control.Monad
import           Data.Proxy
import qualified Data.ByteString.Char8 as C
import           Data.FileEmbed

import           Control.Monad.IO.Class
import qualified Foreign.OpenCL.Bindings as CL
import           GHC.TypeLits

import AI.Funn.SomeNat
import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.CL.Code as C

kSOURCE :: String
kSOURCE = C.unpack $(embedFile "_flat.cl")

reluDiff :: forall s n. (KnownNat n) => Diff (OpenCL s) (Blob s n) (Blob s n)
reluDiff = Diff run
  where
    run xs = do
      ys <- createBlob
      (runKernel kSOURCE "relu"
       [blobArg xs, blobArg ys]
       [] [fromIntegral n] [1])
      return (ys, backward xs)

    backward xs dys = do
      dxs <- createBlob
      (runKernel kSOURCE "relu_back"
       [blobArg xs, blobArg dys, blobArg dxs]
       [] [fromIntegral n] [1])
      return dxs

    n :: Integer
    n = natVal (Proxy :: Proxy n)

sigmoidDiff :: forall s n. (KnownNat n) => Diff (OpenCL s) (Blob s n) (Blob s n)
sigmoidDiff = Diff run
  where
    run xs = do
      ys <- createBlob
      (runKernel sigmoidSrc "run"
       [blobArg xs, blobArg ys]
       [] [fromIntegral n] [1])
      return (ys, backward ys)

    backward ys dys = do
      dxs <- createBlob
      (runKernel sigmoidBackSrc "run"
       [blobArg ys, blobArg dys, blobArg dxs]
       [] [fromIntegral n] [1])
      return dxs

    sigmoidSrc = C.kernel sigmoid
    sigmoid :: ArrayR Float -> ArrayW Float -> CL ()
    sigmoid xs ys = do i <- get_global_id 0
                       z <- eval (exp (xs `at` i))
                       at ys i .= z / (1 + z)

    sigmoidBackSrc = C.kernel sigmoidBack
    sigmoidBack :: ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
    sigmoidBack ys dys dxs = do i <- get_global_id 0
                                let
                                  y = ys `at` i
                                  dy = dys `at` i
                                (dxs `at` i) .= dy * y * (1 - y)

    n :: Integer
    n = natVal (Proxy :: Proxy n)

quadraticCost :: forall s n. (KnownNat n) => Diff (OpenCL s) (Blob s n, Blob s n) Double
quadraticCost = Diff run
  where
    run (xs, ys) = do
      ds <- subBlob xs ys
      rs <- Blob.toList ds
      let o = sum [x^2 | x <- rs]
      return (o, backward ds)

    backward ds δ = do
      dx <- scaleBlob δ ds
      dy <- scaleBlob (-δ) ds
      return (dx, dy)

fcDiff :: forall m n s. (KnownNat m, KnownNat n) =>
          Diff (OpenCL s) (Blob s (m * n + n), Blob s m) (Blob s n)
fcDiff = Diff run
  where
    run (pars, xs) = do
      ys <- createBlob
      (runKernel kSOURCE "fcdiff"
       [int32Arg m, int32Arg n, blobArg pars, blobArg xs, blobArg ys]
       [] [n] [1])
      return (ys, backward pars xs)

    backward :: Blob s (m * n + n) -> Blob s m -> Blob s n ->
                OpenCL s (Blob s (m * n + n), Blob s m)
    backward pars xs dys = do
      dpars <- createBlob
      dxs <- createBlob
      let (dws :: Blob s (m * n), dbs :: Blob s n) = splitBlob dpars
      (runKernel kSOURCE "fcdiff_dws"
       [int32Arg m, int32Arg n, blobArg xs, blobArg dys, blobArg dws]
       [] [m, n] [1, 1])
      return (dpars, dxs)

    m = fromIntegral $ natVal (Proxy :: Proxy m)
    n = fromIntegral $ natVal (Proxy :: Proxy n)
