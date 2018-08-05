module AI.Funn.CL.LazyMem (
  LazyMem, fromStrict, toStrict,
  toStrictFree, clone,
  size, append, slice,
  toChunks
  ) where

import           Control.Applicative
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Monoid
import           Data.Traversable
import           Foreign.Storable (Storable)
import           System.IO.Unsafe

import           AI.Funn.CL.MemSub (MemSub)
import qualified AI.Funn.CL.MemSub as MemSub

-- Mutable buffer backed by MemSub with fast append.
-- Essentially equivalent to a list of MemSub.
data LazyMem a = Empty
               | Leaf {-# UNPACK #-} !Int (MemSub a)
               | Node {-# UNPACK #-} !Int (LazyMem a) (LazyMem a)

instance Semigroup (LazyMem a) where
  (<>) = append

instance Monoid (LazyMem a) where
  mempty = Empty
  mappend = append

-- O(1)
fromStrict :: MemSub a -> LazyMem a
fromStrict mem = Leaf (MemSub.size mem) mem

-- O(size)
toStrict :: (MonadIO m, Storable a) => LazyMem a -> m (MemSub a)
toStrict Empty = MemSub.malloc 0
toStrict (Leaf n mem) = pure mem
toStrict (Node n l r)
  | size l == 0  =  toStrict r
  | size r == 0  =  toStrict l
  | otherwise    =  compact (Node n l r)

-- O(1)
-- Convert to MemSub iff doing so is zero-copy.
toStrictFree :: LazyMem a -> Maybe (MemSub a)
toStrictFree Empty = Nothing
toStrictFree (Leaf n mem) = Just mem
toStrictFree (Node n l r)
  | size l == 0  =  toStrictFree r
  | size r == 0  =  toStrictFree l
  | otherwise    =  Nothing

-- O(n)
toChunks :: LazyMem a -> [MemSub a]
toChunks Empty = []
toChunks (Leaf n mem) = [mem]
toChunks (Node n l r) = toChunks l ++ toChunks r

-- O(size)
clone :: (MonadIO m, Storable a) => LazyMem a -> m (LazyMem a)
clone Empty = pure Empty
clone mem = fromStrict <$> compact mem

-- O(size)
compact :: (MonadIO m, Storable a) => LazyMem a -> m (MemSub a)
compact tree = do
  target <- MemSub.malloc (size tree)
  let go index tree = case tree of
        Empty         -> return ()
        (Leaf n mem)
          | n > 0     -> MemSub.copyInto mem target 0 index n
          | otherwise -> return ()
        (Node n l r)  -> go index l >> go (index + size l) r
  go 0 tree
  return target

-- O(1)
size :: LazyMem a -> Int
size Empty = 0
size (Leaf n _) = n
size (Node n _ _) = n

-- O(1)
append :: LazyMem a -> LazyMem a -> LazyMem a
append Empty r = r
append l Empty = l
append l r = Node (size l + size r) l r

-- O(height)
slice :: Int -> Int -> LazyMem a -> LazyMem a
slice offset len Empty = Empty
slice offset len (Leaf _ mem) = Leaf len (MemSub.slice offset len mem)
slice offset len (Node _ l r)
  | offset >= size l      = slice (offset - size l) len r
  | offset + len < size l = slice offset len l
  | otherwise             = append
                            (slice offset (size l - offset) l)
                            (slice 0 (len - (size l - offset)) r)
