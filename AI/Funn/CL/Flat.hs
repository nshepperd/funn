{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
module AI.Funn.CL.Flat (
  reluDiff, sigmoidDiff, tanhDiff,
  fcDiff, quadraticCost
  ) where

import           Control.Applicative
import           Control.Monad
import           Data.Proxy
import           Debug.Trace

import           Control.Monad.IO.Class
import qualified Foreign.OpenCL.Bindings as CL
import           GHC.TypeLits

import           AI.Funn.SomeNat
import           AI.Funn.Space
import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.CL.Code as C

reluDiff :: forall n m a. (MonadIO m, KnownNat n, Relational a, CLNum a) => Diff m (Blob a n) (Blob a n)
reluDiff = Diff run
  where
    run xs = do
      ys <- relu xs
      return (ys, reluBack xs)

    relu = mapBlob' (\x -> fmax 0 x)
    reluBack = zipWithBlob' (\x dy -> fstep 0 x * dy)

sigmoidDiff :: forall n m a. (MonadIO m, KnownNat n, CLFloating a) => Diff m (Blob a n) (Blob a n)
sigmoidDiff = Diff run
  where
    run xs = do
      ys <- sigmoid xs
      return (ys, sigmoidBack xs)

    sigmoid = mapBlob $ \x -> do
      z <- eval (exp x)
      return $ z / (1 + z)

    sigmoidBack = zipWithBlob $ \x dy -> do
      z <- eval $ exp (-abs x)
      return $ dy * z / (1 + z)^2

tanhDiff :: forall n m a. (MonadIO m, KnownNat n, CLFloating a) => Diff m (Blob a n) (Blob a n)
tanhDiff = Diff run
  where
    run xs = do
      ys <- tanhForward xs
      return (ys, tanhBack xs)

    tanhForward = mapBlob $ \x -> do
      zp <- eval (exp x)
      zm <- eval (exp (-x))
      return $ (zp - zm) / (zp + zm)

    tanhBack = zipWithBlob $ \x dy -> do
      zp <- eval $ exp x
      zm <- eval $ exp (-x)
      z <- eval $ 2 / (zp + zm)
      return $ dy * z^2

quadraticCost :: forall n m a. (MonadIO m, KnownNat n, CLNum a, Floats a) => Diff m (Blob a n, Blob a n) Double
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

fcDiff :: forall α β a m. (MonadIO m, KnownNat α, KnownNat β, CLNum a) =>
          Diff m (Blob a (α * β + β), Blob a α) (Blob a β)
fcDiff = Diff run
  where
    run (pars, xs) = do
      ys <- createBlob
      (runKernel forwardK "run"
       [int32Arg α, int32Arg β, blobArg pars, blobArg xs, blobArg ys]
       [] [β] [1])
      frozen_ys <- unsafeFreeze ys
      return (frozen_ys, backward pars xs)

    backward pars xs dys = do
      dpars <- createBlob
      dxs <- createBlob
      let (dws :: MBlob a (α * β), dbs :: MBlob a β) = splitBlob dpars
      (runKernel backwardwsK "run"
       [int32Arg α, blobArg xs, blobArg dys, blobArg dws]
       [] [α, β] [1, 1])
      (runKernel backwardxsK "run"
       [int32Arg α, int32Arg β, blobArg pars, blobArg dys, blobArg dxs]
       [] [α] [1])
      (runKernel backwardbsK "run"
       [blobArg dys, blobArg dbs]
       [] [β] [1])
      frozen_dpars <- unsafeFreeze dpars
      frozen_dxs <- unsafeFreeze dxs
      return (frozen_dpars, frozen_dxs)

    kahan :: Expr a -> (Expr Int -> Expr a) -> Expr Int -> CL (Expr a)
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
    forwardSrc :: Expr Int -> Expr Int -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    forwardSrc α β pars xs ys = do
      y <- get_global_id 0
      let
        inputs x = (pars `at` (α * y + x)) * (xs `at` x)
      total <- kahan (pars `at` (α*β + y)) inputs α
      at ys y .= total

    backwardwsK = C.kernel backwardwsSrc
    backwardwsSrc :: Expr Int -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    backwardwsSrc α xs dys dws = do
      x <- get_global_id 0
      y <- get_global_id 1
      at dws (y * α + x) .= (xs `at` x) * (dys `at` y)

    backwardxsK = C.kernel backwardxsSrc
    backwardxsSrc :: Expr Int -> Expr Int -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    backwardxsSrc α β pars dys dxs = do
      x <- get_global_id 0
      let
        inputs y = at pars (α * y + x) * at dys y
      total <- kahan 0 inputs β
      at dxs x .= total

    backwardbsK = C.kernel backwardbsSrc
    backwardbsSrc :: ArrayR a -> ArrayW a -> CL ()
    backwardbsSrc dys dbs = do
      y <- get_global_id 0
      at dbs y .= at dys y

    α = fromIntegral $ natVal (Proxy :: Proxy α)
    β = fromIntegral $ natVal (Proxy :: Proxy β)

splitDiff :: forall α β a m. (MonadIO m, KnownNat α, KnownNat β, CLNum a) =>
             Diff m (Blob a (α + β)) (Blob a α, Blob a β)
splitDiff = Diff run
  where
    run ab = pure (splitBlob ab, backward)
    backward (da, db) = catBlob da db

mergeDiff :: forall α β a m. (MonadIO m, KnownNat α, KnownNat β, CLNum a) =>
             Diff m (Blob a α, Blob a β) (Blob a (α + β))
mergeDiff = Diff run
  where
    run (a,b) = do ab <- catBlob a b
                   pure (ab, backward)
    backward dab = pure (splitBlob dab)
