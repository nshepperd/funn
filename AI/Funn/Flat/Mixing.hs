{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 #-}
module AI.Funn.Flat.Mixing (amixDiff) where

import           GHC.TypeLits

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Data.Foldable
import           Data.Monoid
import           Data.Traversable
import           Data.Type.Equality

import           Control.Monad.State.Lazy as State

import           Data.Bits
import           Data.Constraint
import           Data.Proxy
import           Data.Random

import           Control.DeepSeq
import           Numeric.Search.Range

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe
import           Unsafe.Coerce

import           AI.Funn.Common
import           AI.Funn.Flat.Flat
import           AI.Funn.Diff.Diff
import           AI.Funn.NatLog
import           AI.Funn.Flat.Blob (Blob(..), blob, getBlob)
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space

foreign import ccall "layer_resize_forward" resize_forward_ffi :: CInt -> CInt -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_resize_backward" resize_backward_ffi :: CInt -> CInt -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE resize_forward #-}
resize_forward :: Int -> Int -> S.Vector Double -> S.Vector Double
resize_forward a b xs = unsafePerformIO go
  where
    go = do target <- M.replicate b 0 :: IO (M.IOVector Double)
            S.unsafeWith xs $ \sbuf -> do
              M.unsafeWith target $ \tbuf -> do
                resize_forward_ffi (fromIntegral a) (fromIntegral b) sbuf tbuf
            V.unsafeFreeze target

{-# NOINLINE resize_backward #-}
resize_backward :: Int -> Int -> S.Vector Double -> S.Vector Double
resize_backward a b xs = unsafePerformIO go
  where
    go = do target <- M.replicate a 0 :: IO (M.IOVector Double)
            S.unsafeWith xs $ \sbuf -> do
              M.unsafeWith target $ \tbuf -> do
                resize_backward_ffi (fromIntegral a) (fromIntegral b) sbuf tbuf
            V.unsafeFreeze target

type MIX_FFI_TYPE = CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE mix_helper #-}
mix_helper :: MIX_FFI_TYPE -> Int -> Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mix_helper ffi tsize n as bs cs = unsafePerformIO go
  where
    go = do target <- M.replicate tsize 0 :: IO (M.IOVector Double)
            S.unsafeWith as $ \abuf -> do
              S.unsafeWith bs $ \bbuf -> do
                S.unsafeWith cs $ \cbuf -> do
                  M.unsafeWith target $ \tbuf -> do
                    ffi (fromIntegral n) abuf bbuf cbuf tbuf
            V.unsafeFreeze target

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

biasDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n, Blob n) (Blob n)
biasDiff = Diff run
  where
    run (a, b) = do out <- a `plus` b
                    let back d = return (d, d)
                    return (out, back)
------- mixN

resize :: Int -> S.Vector Double -> S.Vector Double
resize n xs
  | V.length xs < n = xs <> V.replicate (n - V.length xs) 0
  | V.length xs > n = V.take n xs
  | otherwise = xs

foreign import ccall "layer_mixN_forward" mixn_forward_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_mixN_backward" mixn_backward_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_mixN_backward_params" mixn_backward_params_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

mixn_forward :: Int -> Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mixn_forward s n = mix_helper mixn_forward_ffi n (s*n)

mixn_backward :: Int -> Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mixn_backward s n = mix_helper mixn_backward_ffi n (s*n)

mixn_backward_params :: Int -> Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mixn_backward_params s n = mix_helper mixn_backward_params_ffi (s*n) (s*n)


mixDiff :: forall proxy s d m. (Monad m, KnownNat s, KnownNat d) =>
           proxy s -> Diff m (Blob (s * (2^d) * d), Blob (2^d)) (Blob (2^d))
mixDiff proxy = Diff run
  where
    run (!pars, !input) = return (blob res, backward)
      where
        par_pieces = V.generate d (\level -> V.slice (level * s * n) (s * n) (getBlob pars))
        (inputs, res) = State.runState (traverse go_forward (V.zip par_pieces table)) (getBlob input)
        backward !delta =
          let (deltas, di) = State.runState (traverseBack go_backward (V.zip par_pieces table)) (getBlob delta)
              dps = V.zipWith3 go_params table inputs deltas
          in return (blob (fold dps), blob di)

    s,d,n :: Int
    s = fromIntegral (natVal (Proxy :: Proxy s))
    d = fromIntegral (natVal (Proxy :: Proxy d))
    n = 2^d

    go_forward :: (S.Vector Double, S.Vector CInt) -> State.State (S.Vector Double) (S.Vector Double)
    go_forward (pars,tab) = do
      input <- get
      let output = mixn_forward s n tab pars input
      put output
      return input

    go_backward :: (S.Vector Double, S.Vector CInt) -> State.State (S.Vector Double) (S.Vector Double)
    go_backward (pars, tab) = do
      delta <- get
      let new = mixn_backward s n tab pars delta
      put new
      return delta

    go_params :: S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
    go_params tab input delta = mixn_backward_params s n tab input delta

    -- table of connected values
    table :: Vector (S.Vector CInt)
    table = V.generate d $ \level ->
      V.fromList . fmap fromIntegral $ do
        i <- [0 .. n-1]
        let f bit = let a = (bit `shiftL` level)
                        part1 = (a .&. (n-1))
                        part2 = (a `xor` part1) `shiftR` d
                    in part1 .|. part2
        fold [[i, i `xor` f bit] | bit <- [0..s-1]]

resizeBlob :: forall a b. (KnownNat b) => Blob a -> Blob b
resizeBlob xs = blob (resize (fromIntegral b) (getBlob xs))
  where
    b = natVal (Proxy :: Proxy b)

resizeDiff :: forall a b m. (Monad m, KnownNat a, KnownNat b) => Diff m (Blob a) (Blob b)
resizeDiff = Diff run
  where
    run a = return (resizeBlob a, pure . resizeBlob)

type Cover a b = 2^(LogCeil (Max a b))

bmixDiff :: forall s a b m. (Monad m, KnownNat s, KnownNat a, KnownNat b) =>
            Proxy s -> Diff m (Blob (s * Cover a b * LogCeil (Max a b)), Blob a) (Blob b)
bmixDiff proxy = second resizeDiff >>> sub >>> resizeDiff
  where
    sub :: Diff m (Blob (s * Cover a b * LogCeil (Max a b)), Blob (Cover a b)) (Blob (Cover a b))
    sub = mixDiff proxy

amixDiff :: forall s a b m. (Monad m, KnownNat s, KnownNat a, KnownNat b) =>
            Proxy s -> Diff m (Blob (s * 2^(LogCeil (Max a b)) * LogCeil (Max a b) + b), Blob a) (Blob b)
amixDiff proxy = first (splitDiff >>> swap) >>> assocR >>> second (bmixDiff proxy) >>> biasDiff
