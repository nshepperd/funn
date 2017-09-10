{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 #-}
module AI.Funn.CL.Mixing (amixDiff) where

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Monad.IO.Class
import           Control.Monad.State.Lazy
import           Data.Foldable
import           Data.List
import           Data.Proxy
import           Data.Traversable
import           Foreign.Storable (Storable)
import           GHC.TypeLits

import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.Code as C
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
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

resize :: (MonadIO m, Storable t, KnownNat a, KnownNat b) => Blob t a -> m (Blob t b)
resize (Blob as) = do
  out@(Blob bs) <- createBlob
  Buffer.copy as bs 0 0 (min (Buffer.size as) (Buffer.size bs))
  Blob.unsafeFreeze out

type MixParams' k d = k * (2^d) * d
type MixParams k a b = MixParams' k (CLog (Max a b)) + b

amixDiff :: forall k α β a m. (MonadIO m, KnownNat k, KnownNat α, KnownNat β, CLNum a)
         => Proxy k -> Diff m (Blob a (MixParams k α β), Blob a α) (Blob a β)
amixDiff proxy = Diff run
  where
    run (pars, input) = do
      input_fit <- resize input
      (output_fit, k) <- runDiff (mixDiff proxy (Proxy @ (CLog (Max α β)))) (sub_par, input_fit)
      output <- resize output_fit
      bs <- addBlob output add
      return (bs, backward k)
      where
        (sub_par, add) = splitBlob pars

    backward k d_output = do
      d_output_fit <- resize d_output
      (dsub_par, d_input_fit) <- k d_output_fit
      d_pars <- catBlob dsub_par d_output
      d_input <- resize d_input_fit
      return (d_pars, d_input)



mixDiff :: forall k d a m proxy. (MonadIO m, KnownNat k, KnownNat d, CLNum a) =>
           proxy k -> proxy d -> Diff m (Blob a (k * (2^d) * d), Blob a (2^d)) (Blob a (2^d))
mixDiff proxy _ = Diff run
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

    slicePars :: BlobT q a (k * (2^d) * d) -> [BlobT q a (k * (2^d))]
    slicePars (Blob buffer) = [Blob (Buffer.slice (k*n*l) (k*n) buffer) | l <- [0..d-1]]

    k,d,n :: Int
    k = fromIntegral (natVal (Proxy :: Proxy k))
    d = fromIntegral (natVal (Proxy :: Proxy d))
    n = 2^d

    go_forward :: (Int, Blob a (k * (2^d))) -> StateT (Blob a (2^d)) m (Blob a (2^d))
    go_forward (l, par) = do
      xs <- get
      ys <- lift createBlob
      lift (runKernel forwardSrc "run"
            [int32Arg n, int32Arg d, int32Arg l, int32Arg k, blobArg par, blobArg xs, blobArg ys]
            [] [fromIntegral n] [])
      frozen_ys <- lift (unsafeFreeze ys)
      put frozen_ys
      return xs

    go_backward :: (Int, Blob a (2^d), Blob a (k * (2^d)), MBlob a (k * (2^d)))
                -> StateT (Blob a (2^d)) m ()
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
