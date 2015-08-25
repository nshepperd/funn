{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.Mixing (freeLayer, biasLayer, hierLayer) where

import           GHC.TypeLits

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Data.Foldable
import           Data.Monoid
import           Data.Traversable

import           Control.Monad.State.Lazy as State

import           Data.Bits
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

import           AI.Funn.Common
import           AI.Funn.Flat
import           AI.Funn.Network
import           AI.Funn.SomeNat

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
                       -- !_ = check "resizeLayer" out (input, a, b)
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
                        -- !_ = check "biasLayer" out (pars, input)
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
                          -- !_ = check "mixLayer" out (pars, input)
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




------ MIX2

foreign import ccall "layer_mix2_forward" mix2_forward_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_mix2_backward" mix2_backward_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "layer_mix2_backward_params" mix2_backward_params_ffi :: CInt -> Ptr CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

mix2_forward :: Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mix2_forward n = mix_helper mix2_forward_ffi n n

mix2_backward :: Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mix2_backward n = mix_helper mix2_backward_ffi n n

mix2_backward_params :: Int -> S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
mix2_backward_params n = mix_helper mix2_backward_params_ffi (2*n) n


traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

hierLayer :: forall a b m. (Monad m, KnownNat a, KnownNat b) => Network m (Blob a) (Blob b)
hierLayer = mix2Layer >>> biasLayer

hierLayer2 :: forall a b m. (Monad m, KnownNat a, KnownNat b) => Network m (Blob a) (Blob b)
hierLayer2 = withNat n (\(Proxy :: Proxy n) -> (mix2Layer :: Network m (Blob a) (Blob n)) >>> mix2Layer >>> biasLayer)
  where
    a,b,d :: Int
    a = fromIntegral (natVal (Proxy :: Proxy a))
    b = fromIntegral (natVal (Proxy :: Proxy b))

    -- Smallest power of 2 containing our inputs and outputs
    Just d = searchFromTo (\d -> 2^d >= max a b) 1 20
    n = fromIntegral (2^d)

mix2Layer :: forall a b m. (Monad m, KnownNat a, KnownNat b) => Network m (Blob a) (Blob b)
mix2Layer = Network eval (2*n*d) initial
  where
    eval pars input = let parameters :: Vector (S.Vector Double)
                          parameters = V.generate d (\level -> V.slice (level * 2 * n) (2 * n) (getParameters pars))
                          input' = resize n (getBlob input)
                          (inputs, res) = State.runState (traverse go_forward (V.zip parameters table)) input'
                          backward delta = let delta' = resize n (getBlob delta)
                                               (deltas, di) = State.runState (traverseBack go_backward (V.zip parameters table)) delta'
                                               dps = V.zipWith3 go_params table inputs deltas
                                           in return (Blob (resize a di), Parameters <$> (V.toList dps))
                      in return (Blob (resize b res), 0, backward)

    initial = Parameters <$> V.replicateM (2*n*d) (normal 0 0.8)

    a,b,d,n :: Int
    a = fromIntegral (natVal (Proxy :: Proxy a))
    b = fromIntegral (natVal (Proxy :: Proxy b))

    -- Smallest power of 2 containing our inputs and outputs
    Just d = searchFromTo (\d -> 2^d >= max a b) 1 20
    n = 2^d

    go_forward :: (S.Vector Double, S.Vector CInt) -> State.State (S.Vector Double) (S.Vector Double)
    go_forward (pars,tab) = do
      input <- get
      let output = mix2_forward n tab pars input
      put output
      return input

    go_backward :: (S.Vector Double, S.Vector CInt) -> State.State (S.Vector Double) (S.Vector Double)
    go_backward (pars, tab) = do
      delta <- get
      let new = mix2_backward n tab pars delta
      put new
      return delta

    go_params :: S.Vector CInt -> S.Vector Double -> S.Vector Double -> S.Vector Double
    go_params tab input delta = mix2_backward_params n tab input delta

    resize :: Int -> S.Vector Double -> S.Vector Double
    resize n xs
      | V.length xs < n = xs <> V.replicate (n - V.length xs) 0
      | V.length xs > n = V.take n xs
      | otherwise = xs

    -- table of connected values
    table :: Vector (S.Vector CInt)
    table = V.generate d $ \level ->
      V.fromList . fmap fromIntegral $ do
        i <- [0 .. n-1]
        [i, i `complementBit` level]
