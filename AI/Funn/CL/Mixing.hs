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
import           Control.Monad.State.Lazy
import           Data.Proxy
import           Data.Foldable
import           Data.List
import           Data.Traversable
import           GHC.TypeLits

import           AI.Funn.SomeNat
import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.CL.Code as C

crossf :: Expr Int -> Expr Int -> Expr Int -> Expr Int -> Expr Int
crossf n d l s = (n-1) .&. (part1 .|. part2)
  where
    part1 = s `shiftL` l
    part2 = s `shiftR` (d - l)

forwardSrc :: String
forwardSrc = C.kernel forwardKernel
forwardKernel :: Expr Int -> Expr Int -> Expr Int -> Expr Int
              -> ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
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
               -> ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
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
                  -> ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
backwardParKernel n d l k xs dys dpars_l = do
  j <- get_global_id 0
  s <- get_global_id 1
  let i = j `xor` crossf n d l s
  at dpars_l (j * k + s) .= at xs i * at dys j

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

traverseBack_ :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f ()
traverseBack_ f = forwards . traverse_ (Backwards . f)

mixDiff :: forall k d s m proxy. (MonadCL s m, KnownNat k, KnownNat d) =>
           proxy k -> Diff m (Blob s (k * (2^d) * d), Blob s (2^d)) (Blob s (2^d))
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
      return (dpars, di)

    slicePars :: Blob s (k * (2^d) * d) -> [Blob s (k * (2^d))]
    slicePars (Blob buffer) = [Blob (Buffer.slice (k*n*l) (k*n) buffer) | l <- [0..d-1]]

    k,d,n :: Int
    k = fromIntegral (natVal (Proxy :: Proxy k))
    d = fromIntegral (natVal (Proxy :: Proxy d))
    n = 2^d

    go_forward :: (Int, Blob s (k * (2^d))) -> StateT (Blob s (2^d)) m (Blob s (2^d))
    go_forward (l, par) = do
      xs <- get
      ys <- lift createBlob
      lift (runKernel forwardSrc "run"
            [int32Arg n, int32Arg d, int32Arg l, int32Arg k, blobArg par, blobArg xs, blobArg ys]
            [] [fromIntegral n] [])
      put ys
      return xs

    go_backward :: (Int, Blob s (2^d), Blob s (k * (2^d)), Blob s (k * (2^d)))
                -> StateT (Blob s (2^d)) m ()
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
      put dxs
