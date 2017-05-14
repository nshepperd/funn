module AI.Funn.CL.Buffer (
  Buffer, malloc, free, arg,
  fromList, toList, slice, concat
  ) where

import           Prelude hiding (concat)

import           Control.Applicative
import           Control.Monad
import           Data.Foldable hiding (toList, concat)
import           Data.List hiding (concat)
import           Data.Monoid
import           Data.Traversable
import qualified Data.Vector.Generic as V

import           Control.Monad.IO.Class
import           Foreign.Storable (Storable)

import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Mem (Mem)
import qualified AI.Funn.CL.Mem as Mem

data Buffer s a = Buffer !(Mem s a) !Int !Int

-- Mem objects have a minimum size of 1. Since we want to support
-- 0-size buffers, we do so here, by allocating with
-- max(1, <buffer size>).

malloc :: (MonadCL s m, Storable a) => Int -> m (Buffer s a)
malloc n = do mem <- Mem.malloc (max 1 n)
              return (Buffer mem 0 n)

free :: MonadIO m => Buffer s a -> m ()
free (Buffer mem _ _) = Mem.free mem

arg :: Buffer s a -> KernelArg s
arg (Buffer mem offset size) = Mem.arg mem <> int32Arg offset

-- O(1)
size :: Buffer s a -> Int
size (Buffer _ _ size) = size

-- O(n)
fromList :: (MonadCL s m, Storable a) => [a] -> m (Buffer s a)
fromList [] = malloc 0
fromList xs = do mem <- Mem.fromList xs
                 return (Buffer mem 0 (length xs))

-- O(n)
toList :: (MonadCL s m, Storable a) => Buffer s a -> m [a]
toList (Buffer mem offset size) = do
  xs <- Mem.peekSubArray offset size mem
  return (V.toList xs)

-- O(1)
slice :: Int -> Int -> Buffer s a -> Buffer s a
slice offset size (Buffer mem _off _size)
  | offset + size <= _size = Buffer mem (_off + offset) size
  | otherwise              = error ("invalid slice in Buffer: "
                                    ++ show offset ++ " + "
                                    ++ show size ++ " > "
                                    ++ show _size)

-- O(n)
concat :: (MonadCL s m, Storable a) => [Buffer s a] -> m (Buffer s a)
concat xs = do
  dst@(Buffer dst_mem _ _) <- malloc totalSize
  for (zip offsets xs) $ \(dst_offset, Buffer src src_offset src_len) -> do
    Mem.copy src dst_mem src_offset dst_offset src_len
  return dst
  where
    totalSize = sum (map size xs)
    offsets = scanl (+) 0 (map size xs)
