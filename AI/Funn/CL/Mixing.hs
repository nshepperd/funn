{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 #-}
module AI.Funn.CL.Mixing (mixDiff) where

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Monad.IO.Class
import           Control.Monad.State.Lazy
import           Data.Foldable
import           Data.List
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits

import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.Code as C
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.SomeNat
import           AI.Funn.TypeLits

crossf :: Expr Int -> Expr Int -> Expr Int -> Expr Int -> Expr Int
crossf n d l s = (n-1) .&. (part1 .|. part2)
  where
    part1 = s `shiftL` l
    part2 = s `shiftR` (d - l)

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

traverseBack_ :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f ()
traverseBack_ f = forwards . traverse_ (Backwards . f)

mixDiff :: forall k d a m proxy. (MonadIO m, KnownNat k, KnownNat d, CLNum a) =>
           proxy k -> Diff m (Blob (k * (2^d) * d) a, Blob (2^d) a) (Blob (2^d) a)
mixDiff proxy = Diff run
  where
    run (pars, input) = do
      let sliced = slicePars pars
      (xs, o) <- runStateT (traverse go_forward (zip [0..] sliced)) input
      return (o, backward xs sliced)

    backward xs sliced dout = do
      dpars <- createBlob
      let dsliced = slicePars dpars
      di <- execStateT (traverseBack_ go_backward (zip4 [0..] xs sliced dsliced)) dout
      frozen_dpars <- unsafeFreeze dpars
      return (frozen_dpars, di)

    slicePars :: BlobT q (k * (2^d) * d) a -> [BlobT q (k * (2^d)) a]
    slicePars (Blob buffer) = [Blob (Buffer.slice (k*n*l) (k*n) buffer) | l <- [0..d-1]]

    k,d,n :: Int
    k = fromIntegral (natVal (Proxy :: Proxy k))
    d = fromIntegral (natVal (Proxy :: Proxy d))
    n = 2^d

    go_forward :: (Int, Blob (k * (2^d)) a) -> StateT (Blob (2^d) a) m (Blob (2^d) a)
    go_forward (l, par) = do
      xs <- get
      ys <- lift createBlob
      lift (runKernel forwardSrc "run"
            [int32Arg n, int32Arg d, int32Arg l, int32Arg k, blobArg par, blobArg xs, blobArg ys]
            [] [fromIntegral n] [])
      frozen_ys <- lift (unsafeFreeze ys)
      put frozen_ys
      return xs

    go_backward :: (Int, Blob (2^d) a, Blob (k * (2^d)) a, MBlob (k * (2^d)) a)
                -> StateT (Blob (2^d) a) m ()
    go_backward (l,xs,par,dpar) = do
      dys <- get
      dxs <- lift createBlob
      lift $ do
        (runKernel backwardSrc "run"
         [int32Arg n, int32Arg d, int32Arg l, int32Arg k, blobArg par, blobArg dys, blobArg dxs]
         [] [fromIntegral n] [])
        (runKernel backwardParSrc "run"
         [int32Arg n, int32Arg d, int32Arg l, int32Arg k, blobArg xs, blobArg dys, blobArg dpar]
         [] [fromIntegral n, fromIntegral k] [])
      frozen_dxs <- lift (unsafeFreeze dxs)
      put frozen_dxs

    forwardSrc :: String
    forwardSrc = C.kernel forwardKernel
    forwardKernel :: Expr Int -> Expr Int -> Expr Int -> Expr Int
                  -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    forwardKernel n d l k pars_l xs ys = do
      j <- get_global_id 0
      parbase <- eval (j * k)
      total <- initvar 0
      forEach 0 k $ \s -> do
        i <- eval (j `xor` crossf n d l s)
        total .= total + (at pars_l (parbase + s)) * (at xs i)
      at ys j .= total

    backwardSrc :: String
    backwardSrc = C.kernel backwardKernel
    backwardKernel :: Expr Int -> Expr Int -> Expr Int -> Expr Int
                   -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    backwardKernel n d l k pars_l dys dxs = do
      i <- get_global_id 0
      total <- initvar 0
      forEach 0 k $ \s -> do
        j <- eval (i `xor` crossf n d l s)
        total .= total + (at pars_l (j * k + s)) * (at dys j)
      at dxs i .= total

    backwardParSrc :: String
    backwardParSrc = C.kernel backwardParKernel
    backwardParKernel :: Expr Int -> Expr Int -> Expr Int -> Expr Int
                      -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    backwardParKernel n d l k xs dys dpars_l = do
      j <- get_global_id 0
      s <- get_global_id 1
      let i = j `xor` crossf n d l s
      at dpars_l (j * k + s) .= at xs i * at dys j
