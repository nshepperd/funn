{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module AI.Funn.CL.Tree (Tree, BufferPure(..), BufferLike(..), up, down, downFree, treeShow, treePieces) where

import           Prelude hiding (concat)

import           Control.Applicative
import           Control.Monad
import           Data.Foldable hiding (toList, concat)
import           Data.List hiding (concat)
import           Data.Monoid
import           Data.Traversable

class BufferPure v where
  size :: v -> Int
  slice :: Int -> Int -> v -> v

class BufferPure v => BufferLike m v where
  malloc :: Int -> m v
  clone :: v -> m v
  concatenate :: [v] -> m v
  copyInto :: v -> v -> Int -> Int -> Int -> m ()

data Tree v = Leaf {-# UNPACK #-} !Int v
            | Node {-# UNPACK #-} !Int (Tree v) (Tree v)

instance BufferPure v => BufferPure (Tree v) where
  size = treeSize
  slice = treeSlice

instance (Monad m, BufferLike m v) => BufferLike m (Tree v) where
  malloc = treeMalloc
  clone = treeClone
  concatenate = return . treeConcat
  copyInto = treeCopyInto

indent :: String -> String
indent = unlines . map ("  "++) . lines

treeShow :: Tree v -> String
treeShow (Leaf n _) = "Leaf <" ++ show n ++ ">"
treeShow (Node n l r) = "Node <" ++ show n ++ ">\n" ++ indent (treeShow l ++ "\n" ++ treeShow r)

-- O(1)
up :: BufferPure v => v -> Tree v
up v = Leaf (size v) v

-- O(size)
down :: (Monad m, BufferLike m v) => Tree v -> m v
down t = do (Leaf _ v) <- compact t
            return v

-- O(1)
downFree :: Tree v -> Maybe v
downFree (Leaf _ v) = Just v
downFree _          = Nothing

treePieces :: Tree v -> [(Int, v)]
treePieces tree = go tree 0
  where
    go (Leaf _ v) off = [(off, v)]
    go (Node _ l r) off = go l off ++ go r (off + treeSize l)

-- O(1)
treeSize :: Tree v -> Int
treeSize (Leaf n _) = n
treeSize (Node n _ _) = n

-- O(1) ish
treeMalloc :: (Functor m, BufferLike m v) => Int -> m (Tree v)
treeMalloc sz = Leaf sz <$> malloc sz

-- O(size)
treeClone :: (Monad m, BufferLike m v) => Tree v -> m (Tree v)
treeClone (Leaf n v) = Leaf n <$> clone v
treeClone tree       = compact tree

-- O(1) ish
treeSlice :: (BufferPure v) => Int -> Int -> Tree v -> Tree v
treeSlice !offset sz (Leaf _ v) = Leaf sz (slice offset sz v)
treeSlice !offset sz t@(Node _ l r)
  | offset + sz <= size l   = treeSlice offset sz l
  | offset      >= size l   = treeSlice (offset - size l) sz r
  | otherwise               = let lpart = treeSlice offset (size l - offset) l
                                  rpart = treeSlice 0 (sz - (size l - offset)) r
                              in Node sz lpart rpart

-- O(1) ish
treeConcat :: [Tree v] -> Tree v
treeConcat [] = error "empty"
treeConcat [t] = t
treeConcat (s:t:xs) = treeConcat (Node (treeSize s + treeSize t) s t : xs)

-- O(len)
treeCopyInto :: (Monad m, BufferLike m v) => Tree v -> Tree v -> Int -> Int -> Int -> m ()
treeCopyInto (Leaf _ srcBuffer) (Leaf _ dstBuffer) srcOffset dstOffset len = do
  copyInto srcBuffer dstBuffer srcOffset dstOffset len
treeCopyInto src (Node _ dstL dstR) srcOffset dstOffset len
  | dstOffset + len <= size dstL    = treeCopyInto src dstL srcOffset dstOffset len
  | dstOffset       >= size dstL    = treeCopyInto src dstR srcOffset (dstOffset - size dstL) len
  | otherwise                       = do treeCopyInto src dstL srcOffset dstOffset lenL
                                         treeCopyInto src dstR (srcOffset + lenL) 0 (len - lenL)
                                           where
                                             lenL = size dstL - dstOffset
treeCopyInto (Node _ srcL srcR) dst srcOffset dstOffset len
  | srcOffset + len <= size srcL   = treeCopyInto srcL dst srcOffset dstOffset len
  | srcOffset       >= size srcL   = treeCopyInto srcR dst (srcOffset - size srcL) dstOffset len
  | otherwise                      = do treeCopyInto srcL dst srcOffset dstOffset lenL
                                        treeCopyInto srcR dst 0 (dstOffset + lenL) (len - lenL)
                                          where
                                            lenL = size srcL - srcOffset

-- O(size)
compact :: (Monad m, BufferLike m v) => Tree v -> m (Tree v)
compact (Leaf n v) = return (Leaf n v)
compact tree       = do
  target <- malloc (size tree)
  go target 0 tree
  return (Leaf (size tree) target)
  where
    go dst dstOffset t = case t of
      (Leaf n v) -> copyInto v dst 0 dstOffset n
      (Node _ l r) -> do go dst dstOffset l
                         go dst (dstOffset + size l) r
