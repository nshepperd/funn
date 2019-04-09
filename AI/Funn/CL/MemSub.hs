{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module AI.Funn.CL.MemSub (
  MemSub, malloc, free, arg,
  fromVector, toVector,
  fromList, toList, size,
  slice, concatenate, clone, copyInto
  ) where


import           Control.Applicative
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable hiding (toList)
import           Data.List
import           Data.Monoid
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           Foreign.Storable (Storable)

import           AI.Funn.CL.Mem (Mem)
import qualified AI.Funn.CL.Mem as Mem
import           AI.Funn.CL.MonadCL

-- A MemSub is a view into a memory object, supporting efficient
-- slicing and conversion, comparable to a storable Vector.

data MemSub a = MemSub !(Mem a) !Int !Int

-- O(1)
size :: MemSub a -> Int
size (MemSub _ _ size) = size

-- O(1)
slice :: Int -> Int -> MemSub a -> MemSub a
slice offset size (MemSub mem _off _size)
  | offset + size <= _size = MemSub mem (_off + offset) size
  | otherwise              = error ("invalid slice in MemSub: "
                                    ++ show offset ++ " + "
                                    ++ show size ++ " > "
                                    ++ show _size)

-- Mem objects have a minimum size of 1. Since we want to support
-- 0-size buffers, we do so here, by allocating with
-- max(1, <buffer size>).

-- O(1)
malloc :: (MonadIO m, Storable a) => Int -> m (MemSub a)
malloc n = do mem <- Mem.malloc (max 1 n)
              return (MemSub mem 0 n)

-- O(size)
clone :: (MonadIO m, Storable a) => (MemSub a) -> m (MemSub a)
clone buffer = concatenate [buffer]

-- O(total size)
concatenate :: (MonadIO m, Storable a) => [MemSub a] -> m (MemSub a)
concatenate xs = do
  dst@(MemSub dst_mem _ _) <- malloc totalSize
  for_ (zip offsets xs) $ \(dst_offset, MemSub src src_offset src_len) -> do
    Mem.copy src dst_mem src_offset dst_offset src_len
  return dst
  where
    totalSize = sum (map size xs)
    offsets = scanl (+) 0 (map size xs)

-- O(size src)
copyInto :: (MonadIO m, Storable a) => MemSub a -> MemSub a -> Int -> Int -> Int -> m ()
copyInto (MemSub src srcOff _) (MemSub dst dstOff _) srcOffset dstOffset len =
  Mem.copy src dst srcOffsetTotal dstOffsetTotal len
  where
    srcOffsetTotal = srcOff + srcOffset
    dstOffsetTotal = dstOff + dstOffset

-- Extra functions

free :: MonadIO m => MemSub a -> m ()
free (MemSub mem _ _) = Mem.free mem

arg :: MemSub a -> KernelArg
arg (MemSub mem offset size) = Mem.arg mem <> int32Arg offset

-- O(n)
fromVector :: (MonadIO m, Storable a) => S.Vector a -> m (MemSub a)
fromVector xs = do buf <- malloc (V.length xs)
                   let (MemSub mem 0 _) = buf
                   Mem.pokeSubArray 0 xs mem
                   return buf

fromList :: (MonadIO m, Storable a) => [a] -> m (MemSub a)
fromList xs = fromVector (S.fromList xs)

-- O(n)
toVector :: (MonadIO m, Storable a) => MemSub a -> m (S.Vector a)
toVector (MemSub mem offset size) = Mem.peekSubArray offset size mem

toList :: (MonadIO m, Storable a) => MemSub a -> m [a]
toList buffer = S.toList <$> toVector buffer
