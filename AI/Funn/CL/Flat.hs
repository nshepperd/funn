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
import           Debug.Trace

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

reluDiff :: forall n s. (KnownNat n) => Diff (OpenCL s) (Blob s n) (Blob s n)
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

sigmoidDiff :: forall n s. (KnownNat n) => Diff (OpenCL s) (Blob s n) (Blob s n)
sigmoidDiff = Diff run
  where
    run xs = do
      ys <- createBlob
      (runKernel sigmoidSrc "run"
       [blobArg xs, blobArg ys]
       [] [fromIntegral n] [1])
      return (ys, backward xs)

    backward xs dys = do
      dxs <- createBlob
      (runKernel sigmoidBackSrc "run"
       [blobArg xs, blobArg dys, blobArg dxs]
       [] [fromIntegral n] [])
      return dxs

    sigmoidSrc = C.kernel sigmoid
    sigmoid :: ArrayR Float -> ArrayW Float -> CL ()
    sigmoid xs ys = do i <- get_global_id 0
                       z <- eval (exp (xs `at` i))
                       at ys i .= z / (1 + z)

    sigmoidBackSrc = C.kernel sigmoidBack
    sigmoidBack :: ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
    sigmoidBack xs dys dxs = do i <- get_global_id 0
                                let
                                  x = xs `at` i
                                  dy = dys `at` i
                                z <- eval $ exp (-abs x)
                                (dxs `at` i) .= dy * z / (1 + z)^2

    n :: Integer
    n = natVal (Proxy :: Proxy n)

quadraticCost :: forall n s. (KnownNat n) => Diff (OpenCL s) (Blob s n, Blob s n) Double
quadraticCost = Diff run
  where
    run (xs, ys) = do
      ds <- subBlob xs ys
      rs <- Blob.toList ds
      let o = sum [x^2 | x <- rs]
      return (o, backward ds)

    backward ds δ = do
      dx <- scaleBlob (2 * δ) ds
      dy <- scaleBlob (-2 * δ) ds
      return (dx, dy)

fcDiff :: forall α β m s. (MonadCL s m, KnownNat α, KnownNat β) =>
          Diff m (Blob s (α * β + β), Blob s α) (Blob s β)
fcDiff = Diff run
  where
    run (pars, xs) = do
      ys <- createBlob
      (runKernel forwardK "run"
       [int32Arg α, int32Arg β, blobArg pars, blobArg xs, blobArg ys]
       [] [β] [1])
      return (ys, backward pars xs)

    backward :: Blob s (α * β + β) -> Blob s α -> Blob s β ->
                m (Blob s (α * β + β), Blob s α)
    backward pars xs dys = do
      dpars <- createBlob
      dxs <- createBlob
      let (dws :: Blob s (α * β), dbs :: Blob s β) = splitBlob dpars
      (runKernel backwardwsK "run"
       [int32Arg α, blobArg xs, blobArg dys, blobArg dws]
       [] [α, β] [1, 1])
      (runKernel backwardxsK "run"
       [int32Arg α, int32Arg β, blobArg pars, blobArg dys, blobArg dxs]
       [] [α] [1])
      (runKernel backwardbsK "run"
       [blobArg dys, blobArg dbs]
       [] [β] [1])
      return (dpars, dxs)

    kahan :: Expr Float -> (Expr Int -> Expr Float) -> Expr Int -> CL (Expr Float)
    kahan initial inputs count = do
      total <- eval initial
      comp <- eval 0
      forEach 0 count $ \x -> do
        input <- eval (inputs x)
        add <- eval (input - comp)
        t <- eval (total + add)
        comp .= (t - total) - add;
        total .= t;
      return total

    forwardK = C.kernel forwardSrc
    forwardSrc :: Expr Int -> Expr Int -> ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
    forwardSrc α β pars xs ys = do
      y <- get_global_id 0
      let
        inputs x = (pars `at` (α * y + x)) * (xs `at` x)
      total <- kahan (pars `at` (α*β + y)) inputs α
      at ys y .= total

    backwardwsK = C.kernel backwardwsSrc
    backwardwsSrc :: Expr Int -> ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
    backwardwsSrc α xs dys dws = do
      x <- get_global_id 0
      y <- get_global_id 1
      at dws (y * α + x) .= (xs `at` x) * (dys `at` y)

    backwardxsK = C.kernel backwardxsSrc
    backwardxsSrc :: Expr Int -> Expr Int -> ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
    backwardxsSrc α β pars dys dxs = do
      x <- get_global_id 0
      let
        inputs y = at pars (α * y + x) * at dys y
      total <- kahan 0 inputs β
      at dxs x .= total

    backwardbsK = C.kernel backwardbsSrc
    backwardbsSrc :: ArrayR Float -> ArrayW Float -> CL ()
    backwardbsSrc dys dbs = do
      y <- get_global_id 0
      at dbs y .= at dys y

    α = fromIntegral $ natVal (Proxy :: Proxy α)
    β = fromIntegral $ natVal (Proxy :: Proxy β)
