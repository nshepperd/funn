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

import           Control.Monad.IO.Class
import           Foreign.Storable (Storable)

import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Mem (Mem)
import qualified AI.Funn.CL.Mem as Mem

data Buffer s a = Buffer !(Mem s a) !Int !Int

malloc :: Storable a => Int -> OpenCL s (Buffer s a)
malloc n = do mem <- Mem.malloc n
              return (Buffer mem 0 n)

free :: MonadIO m => Buffer s a -> m ()
free (Buffer mem _ _) = Mem.free mem

arg :: Buffer s a -> KernelArg s
arg (Buffer mem offset size) = Mem.arg mem <> int32Arg offset

fromList :: Storable a => [a] -> OpenCL s (Buffer s a)
fromList xs = do mem <- Mem.fromList xs
                 return (Buffer mem 0 (length xs))

-- Inefficient!
toList :: (Storable a) => Buffer s a -> OpenCL s [a]
toList (Buffer mem offset size) = do xs <- Mem.toList mem
                                     return (take size $ drop offset $ xs)

slice :: Int -> Int -> Buffer s a -> Buffer s a
slice offset size (Buffer mem _off _size)
  | offset + size <= _size = Buffer mem (_off + offset) size
  | otherwise              = error ("invalid slice in Buffer: "
                                    ++ show offset ++ " + "
                                    ++ show size ++ " > "
                                    ++ show _size)

concat :: Storable a => [Buffer s a] -> OpenCL s (Buffer s a)
concat xs = do values <- traverse toList xs
               fromList (fold values)
