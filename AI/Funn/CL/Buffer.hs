{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
module AI.Funn.CL.Buffer (
  Buffer, malloc, arg,
  fromVector, toVector,
  fromList, toList, size,
  slice, concat, clone, copy,
  addInto
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
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tree (Tree)
import qualified AI.Funn.CL.Tree as Tree
import           AI.Funn.Space

data Buffer a = Buffer (IORef (Tree (MemSub a)))

malloc :: (MonadIO m, Storable a) => Int -> m (Buffer a)
malloc n = Tree.malloc n >>= fromTree

-- free :: MonadIO m => Buffer a -> m ()
-- free (Buffer mem _ _) = Mem.free mem

fromTree :: MonadIO m => Tree (MemSub a) -> m (Buffer a)
fromTree tree = Buffer <$> liftIO (newIORef tree)

getTree :: MonadIO m => Buffer a -> m (Tree (MemSub a))
getTree (Buffer ioref) = liftIO (readIORef ioref)

compact :: (HasCallStack, Storable a) => Buffer a -> MemSub a
compact (Buffer ioref) = unsafePerformIO $ do
  tree <- readIORef ioref
  case Tree.downFree tree of
    Just mem -> return mem
    Nothing  -> do mem <- Tree.down tree
                   -- putStrLn $ "compacting " ++ Tree.treeShow tree
                   -- putStrLn $ "at " ++ prettyCallStack callStack
                   writeIORef ioref (Tree.up mem)
                   return mem

arg :: (HasCallStack, Storable a) => Buffer a -> KernelArg
arg buffer = MemSub.arg (compact buffer)

-- O(1)
size :: Buffer a -> Int
size buffer = unsafePerformIO (Tree.size <$> getTree buffer)

-- O(n)
fromVector :: (MonadIO m, Storable a) => S.Vector a -> m (Buffer a)
fromVector xs = fromTree . Tree.up =<< MemSub.fromVector xs

fromList :: (MonadIO m, Storable a) => [a] -> m (Buffer a)
fromList xs = fromTree . Tree.up =<< MemSub.fromList xs

-- O(n)
toVector :: (MonadIO m, Storable a) => Buffer a -> m (S.Vector a)
toVector buffer = MemSub.toVector (compact buffer)

toList :: (MonadIO m, Storable a) => Buffer a -> m [a]
toList buffer = MemSub.toList (compact buffer)

-- O(1)
slice :: Int -> Int -> Buffer a -> Buffer a
slice offset size buffer = unsafePerformIO $ do
  tree <- getTree buffer
  fromTree (Tree.slice offset size tree)

clone :: (MonadIO m, Storable a) => (Buffer a) -> m (Buffer a)
clone buffer = do
  tree <- getTree buffer
  tree' <- Tree.clone tree
  fromTree tree'

-- O(n)
concat :: (MonadIO m, Storable a) => [Buffer a] -> m (Buffer a)
concat xs = do
  trees <- traverse getTree xs
  tree <- Tree.concatenate trees
  fromTree tree

copy :: (MonadIO m, Storable a) => Buffer a -> Buffer a -> Int -> Int -> Int -> m ()
copy src dst srcOffset dstOffset len = do
  srcTree <- getTree src
  dstTree <- getTree dst
  Tree.copyInto srcTree dstTree srcOffset dstOffset len

memoTable :: KTable Precision
memoTable = newKTable unsafePerformIO

addInto :: (MonadIO m, CLFloats a) => Buffer a -> Buffer a -> m ()
addInto src dst = do
  srcTree <- getTree src
  let dstSub = compact dst
  traverse_ (go dstSub) (Tree.treePieces srcTree)
  where
    go dstSub (offset, src) = addMemInto src (MemSub.slice offset (MemSub.size src) dstSub)

addMemInto :: forall a m. (MonadIO m, CLFloats a) => MemSub a -> MemSub a -> m ()
addMemInto src dst = case addSrc of
                       KernelProgram k -> runCompiled k [MemSub.arg src, MemSub.arg dst] [] [fromIntegral (MemSub.size src)] []
  where
    addSrc :: KernelProgram '[ArrayR a, ArrayW a]
    addSrc = memo memoTable (precision @a) $ compile $ \xs zs -> do
      i <- get_global_id 0
      at zs i .= at zs i + at xs i
