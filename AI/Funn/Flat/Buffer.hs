{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
module AI.Funn.Flat.Buffer (Buffer, getVector, fromVector, sumBuffers) where

import           Control.Applicative
import           Control.Exception
import           Data.Foldable hiding (toList)
import qualified Data.Foldable as F
import           Data.Traversable
import           Data.Monoid
import           Data.Proxy
import           Data.Random

import           Control.DeepSeq
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe
import           Data.IORef

data Buffer = Buffer !Int (IORef [S.Vector Double])

instance Show Buffer where
  show buf = show (getVector buf)

instance Read Buffer where
  readsPrec n s = map (\(a, xs) -> (fromVector a, xs)) (readsPrec n s)

instance NFData Buffer where
  rnf (Buffer n ref) = unsafePerformIO $ do
    vs <- readIORef ref
    return (rnf vs)

instance Monoid Buffer where
  mempty = empty_buffer
  mappend = append

compact :: IORef [S.Vector Double] -> IO (S.Vector Double)
compact ref = do
  vs <- readIORef ref
  let v = fold vs
  writeIORef ref [v]
  return v

getVector :: Buffer -> S.Vector Double
getVector (Buffer n ref) = unsafePerformIO $ do
  vs <- readIORef ref
  case vs of
    []   -> pure V.empty
    [xs] -> pure xs
    _    -> compact ref

fromVector :: S.Vector Double -> Buffer
fromVector v = unsafePerformIO $ do
  ref <- newIORef [v]
  return (Buffer (V.length v) ref)

append :: Buffer -> Buffer -> Buffer
append (Buffer a one) (Buffer b two) = unsafePerformIO $ do
  xs <- readIORef one
  ys <- readIORef two
  Buffer (a+b) <$> newIORef (xs ++ ys)

empty_buffer :: Buffer
empty_buffer = unsafePerformIO $ (Buffer 0 <$> newIORef [])

foreign import ccall "vector_add" ffi_vector_add :: CInt -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE vector_add #-}
vector_add :: M.IOVector Double -> S.Vector Double -> IO ()
vector_add tgt src = do M.unsafeWith tgt $ \tbuf -> do
                          S.unsafeWith src $ \sbuf -> do
                            ffi_vector_add (fromIntegral n) tbuf sbuf
  where
    n = V.length src

getList :: Buffer -> IO [S.Vector Double]
getList (Buffer n ref) = readIORef ref

getSize :: Buffer -> Int
getSize (Buffer n _) = n

addConcatInto :: M.IOVector Double -> [S.Vector Double] -> IO ()
addConcatInto target [] = return ()
addConcatInto target (v:vs) = do
  assert (V.length v <= M.length target) (return ())
  vector_add target v
  addConcatInto (M.drop (V.length v) target) vs

addBuffersInto :: M.IOVector Double -> [Buffer] -> IO ()
addBuffersInto target [] = return ()
addBuffersInto target (b:bs) = do
  addConcatInto target =<< getList b
  addBuffersInto target bs

sumBuffers :: [Buffer] -> Buffer
sumBuffers [x] = x
sumBuffers xs = unsafePerformIO $ do
  target <- M.replicate n 0
  addBuffersInto target xs
  fromVector <$> V.unsafeFreeze target
  where
    n = minimum (map getSize xs)

slice :: Int -> Int -> Buffer -> Buffer
slice off size buffer = fromVector $ V.slice off size (getVector buffer)
