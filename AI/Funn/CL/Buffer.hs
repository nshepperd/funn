module AI.Funn.CL.Buffer (
  Buffer, malloc, free, arg,
  fromVector, toVector,
  fromList, toList,
  slice, concat, clone
  ) where

import           Prelude hiding (concat)

import           Control.Applicative
import           Control.Monad
import           Data.Foldable hiding (toList, concat)
import           Data.List hiding (concat)
import           Data.Monoid
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S

import           Control.Monad.IO.Class
import           Foreign.Storable (Storable)

import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Mem (Mem)
import qualified AI.Funn.CL.Mem as Mem

-- A Buffer is a view into a memory object, supporting efficient
-- slicing and conversion, comparable to a storable Vector.

data Buffer a = Buffer !(Mem a) !Int !Int

-- Mem objects have a minimum size of 1. Since we want to support
-- 0-size buffers, we do so here, by allocating with
-- max(1, <buffer size>).

malloc :: (MonadIO m, Storable a) => Int -> m (Buffer a)
malloc n = do mem <- Mem.malloc (max 1 n)
              return (Buffer mem 0 n)

free :: MonadIO m => Buffer a -> m ()
free (Buffer mem _ _) = Mem.free mem

arg :: Buffer a -> KernelArg
arg (Buffer mem offset size) = Mem.arg mem <> int32Arg offset

-- O(1)
size :: Buffer a -> Int
size (Buffer _ _ size) = size

-- O(n)
fromVector :: (MonadIO m, Storable a) => S.Vector a -> m (Buffer a)
fromVector xs = do buf@(Buffer mem 0 _) <- malloc (V.length xs)
                   Mem.pokeSubArray 0 xs mem
                   return buf

fromList :: (MonadIO m, Storable a) => [a] -> m (Buffer a)
fromList xs = fromVector (S.fromList xs)

-- O(n)
toVector :: (MonadIO m, Storable a) => Buffer a -> m (S.Vector a)
toVector (Buffer mem offset size) = Mem.peekSubArray offset size mem

toList :: (MonadIO m, Storable a) => Buffer a -> m [a]
toList buffer = S.toList <$> toVector buffer

-- O(1)
slice :: Int -> Int -> Buffer a -> Buffer a
slice offset size (Buffer mem _off _size)
  | offset + size <= _size = Buffer mem (_off + offset) size
  | otherwise              = error ("invalid slice in Buffer: "
                                    ++ show offset ++ " + "
                                    ++ show size ++ " > "
                                    ++ show _size)

clone :: (MonadIO m, Storable a) => (Buffer a) -> m (Buffer a)
clone buffer = concat [buffer]

-- O(n)
concat :: (MonadIO m, Storable a) => [Buffer a] -> m (Buffer a)
concat xs = do
  dst@(Buffer dst_mem _ _) <- malloc totalSize
  for_ (zip offsets xs) $ \(dst_offset, Buffer src src_offset src_len) -> do
    Mem.copy src dst_mem src_offset dst_offset src_len
  return dst
  where
    totalSize = sum (map size xs)
    offsets = scanl (+) 0 (map size xs)
