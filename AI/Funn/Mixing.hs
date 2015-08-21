{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.Mixing (freeLayer, biasLayer) where

import           GHC.TypeLits

import           Control.Applicative
import           Data.Foldable
import           Data.Traversable
import           Data.Monoid
import           Data.Proxy
import           Data.Random

import           Control.DeepSeq

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe

import           AI.Funn.Network
import           AI.Funn.Flat

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

resizeLayer :: forall a b m. (Monad m, KnownNat a, KnownNat b) => Network m (Blob a) (Blob b)
resizeLayer = Network eval 0 (pure mempty)
  where
    eval _ input = let out = resize_forward a b (getBlob input)
                       backward delta = let δ = resize_backward a b (getBlob delta)
                                        in return (Blob δ, [])
                   in return (Blob out, 0, backward)

    a,b :: Int
    a = fromIntegral (natVal (Proxy :: Proxy a))
    b = fromIntegral (natVal (Proxy :: Proxy b))

type MIX_FFI_TYPE = CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_mix_forward" mix_forward_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_mix_backward" mix_backward_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_mix_backward_params" mix_backward_params_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

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

mix_forward :: Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mix_forward n = mix_helper mix_forward_ffi n n

mix_backward :: Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mix_backward n = mix_helper mix_backward_ffi n n

mix_backward_params :: Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mix_backward_params n = mix_helper mix_backward_params_ffi (3*n) n

multiMixLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
multiMixLayer = go p
  where
    go 1 = mixLayer
    go x = mixLayer >>> (go (x-1))

    n :: Int
    n = fromIntegral (natVal (Proxy :: Proxy n))

    p :: Int
    p = max 1 (floor $ logBase 2 (fromIntegral n))

freeLayer :: forall a b m. (Monad m, KnownNat a, KnownNat b) => Network m (Blob a) (Blob b)
freeLayer
  | a > b = multiMixLayer >>> resizeLayer
  | a <= b = resizeLayer >>> multiMixLayer
  where
    a,b :: Int
    a = fromIntegral (natVal (Proxy :: Proxy a))
    b = fromIntegral (natVal (Proxy :: Proxy b))

biasLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
biasLayer = Network ev n initial
  where
    ev pars input = let out = Blob (getBlob input + getParameters pars)
                        backward δ = return (δ, [Parameters (getBlob δ)])
                    in return (out, 0, backward)

    initial = pure (Parameters (V.replicate n 0))

    n :: Int
    n = fromIntegral (natVal (Proxy :: Proxy n))

mixLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
mixLayer = Network eval (3*n) initial
  where
    eval pars input = let out = mix_forward n table (getParameters pars) (getBlob input)
                          backward delta = let di = mix_backward n table (getParameters pars) (getBlob delta)
                                               dp = mix_backward_params n table (getBlob input) (getBlob delta)
                                           in return (Blob di, [Parameters dp])
                      in return (Blob out, 0, backward)

    initial = Parameters <$> V.replicateM (3*n) (normal 0 0.5)

    n :: Int
    n = fromIntegral (natVal (Proxy :: Proxy n))

    -- table of connected values
    table :: S.Vector CInt
    table = V.convert $ fromIntegral <$>
            do i <- pointing
               V.fromList [(i - 1) `mod` n, i, (i + 1) `mod` n]

    pointing :: Vector Int
    pointing = shuffle (V.generate n id)

    shuffle :: Vector a -> Vector a
    shuffle vs = part1 <> part2
      where
        part1 = V.generate (V.length vs `div` 2) (\i -> vs V.! (i*2+1))
        part2 = V.generate ((V.length vs + 1) `div` 2) (\i -> vs V.! (i*2))
