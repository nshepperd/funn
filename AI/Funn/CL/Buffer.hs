{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module AI.Funn.CL.Buffer (
  Buffer, malloc, arg,
  fromVector, toVector,
  fromList, toList, size,
  slice, concat, clone, copyInto,
  addInto, fromMemSub, toMemSub
  ) where

import           Prelude hiding (concat)

import           Control.Applicative
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable hiding (toList, concat)
import           Data.IORef
import           Data.List hiding (concat)
import           Data.Monoid
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           Foreign.Storable (Storable)
import           GHC.Stack
import           System.IO.Unsafe

import           AI.Funn.CL.DSL.Code
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.Function
import           AI.Funn.CL.MemSub (MemSub)
import qualified AI.Funn.CL.MemSub as MemSub
import           AI.Funn.CL.LazyMem (LazyMem)
import qualified AI.Funn.CL.LazyMem as LazyMem
import           AI.Funn.CL.MonadCL
import           AI.Funn.Space

newtype Buffer a = Buffer (IORef (LazyMem a))

malloc :: (MonadIO m, Storable a) => Int -> m (Buffer a)
malloc n = fromMemSub <$> MemSub.malloc n

fromLazyMem :: LazyMem a -> Buffer a
fromLazyMem mem = unsafePerformIO (Buffer <$> newIORef mem)

toLazyMem :: MonadIO m => Buffer a -> m (LazyMem a)
toLazyMem (Buffer ioref) = liftIO (readIORef ioref)

fromMemSub :: MemSub a -> Buffer a
fromMemSub mem = fromLazyMem (LazyMem.fromStrict mem)

toMemSub :: (MonadIO m, Storable a) => Buffer a -> m (MemSub a)
toMemSub (Buffer ioref) = liftIO $ do
  lm <- readIORef ioref
  case LazyMem.toStrictFree lm of
    Just mem -> return mem
    Nothing -> do
      mem <- LazyMem.toStrict lm
      writeIORef ioref (LazyMem.fromStrict mem)
      return mem

arg :: (Storable a) => Buffer a -> KernelArg
arg buffer = MemSub.arg (unsafePerformIO $ toMemSub buffer)

-- O(1)
size :: Buffer a -> Int
size buffer = unsafePerformIO (LazyMem.size <$> toLazyMem buffer)

-- O(size)
fromVector :: (MonadIO m, Storable a) => S.Vector a -> m (Buffer a)
fromVector xs = fromMemSub <$> MemSub.fromVector xs

fromList :: (MonadIO m, Storable a) => [a] -> m (Buffer a)
fromList xs = fromMemSub <$> MemSub.fromList xs

-- O(size)
toVector :: (MonadIO m, Storable a) => Buffer a -> m (S.Vector a)
toVector buffer = toMemSub buffer >>= MemSub.toVector

toList :: (MonadIO m, Storable a) => Buffer a -> m [a]
toList buffer = toMemSub buffer >>= MemSub.toList

-- O(1)
slice :: Int -> Int -> Buffer a -> Buffer a
slice offset len buffer = unsafePerformIO $ do
  lm <- toLazyMem buffer
  return (fromLazyMem (LazyMem.slice offset len lm))

clone :: (MonadIO m, Storable a) => (Buffer a) -> m (Buffer a)
clone buffer = do
  lm <- toLazyMem buffer
  fromLazyMem <$> LazyMem.clone lm

-- O(n)
concat :: [Buffer a] -> Buffer a
concat xs = unsafePerformIO $ do
  lms <- traverse toLazyMem xs
  return (fromLazyMem (fold lms))

copyInto :: (MonadIO m, Storable a) => Buffer a -> Buffer a -> Int -> Int -> Int -> m ()
copyInto src dst srcOffset dstOffset len = do
  srcSub <- toMemSub src
  dstSub <- toMemSub dst
  MemSub.copyInto srcSub dstSub srcOffset dstOffset len

memoTable :: KTable Precision
memoTable = newKTable unsafePerformIO

addInto :: (MonadIO m, CLFloats a) => Buffer a -> Buffer a -> m ()
addInto src dst = do
  srcLM <- toLazyMem src
  dstSub <- toMemSub dst
  traverse_ (go dstSub) (parts 0 $ LazyMem.toChunks srcLM)
  where
    parts n [] = []
    parts n (sub:xs) = (n,sub) : parts (n + MemSub.size sub) xs
    go dstSub (offset, src) = addMemInto src (MemSub.slice offset (MemSub.size src) dstSub)

addMemInto :: forall a m. (MonadIO m, CLFloats a) => MemSub a -> MemSub a -> m ()
addMemInto src dst = case addSrc of
                       KernelProgram k -> runCompiled k [MemSub.arg src, MemSub.arg dst] [] [fromIntegral (MemSub.size src)] []
  where
    addSrc :: KernelProgram '[ArrayR a, ArrayW a]
    addSrc = memo memoTable (precision @a) $ compile $ \xs zs -> do
      i <- get_global_id 0
      at zs i .= at zs i + at xs i
