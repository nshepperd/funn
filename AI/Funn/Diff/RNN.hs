{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE PartialTypeSignatures #-}
module AI.Funn.Diff.RNN (scanlDiff, mapDiff, zipDiff, unzipDiff, vsumDiff) where

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Monad.State.Lazy
import           Data.Foldable
import           Data.Traversable

import           Data.Coerce
import           Debug.Trace

import           Foreign.C
import           Foreign.Ptr
import           System.IO
import           System.IO.Unsafe

import           Data.Functor.Identity

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M
import qualified Numeric.LinearAlgebra.HMatrix as HM

import           AI.Funn.Diff.Diff (Derivable(..), Diff(..), Additive(..))
import qualified AI.Funn.Diff.Diff as Diff

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

scanlDiff :: forall m x s i o. (Monad m, Additive m (D x)) =>
             Diff m (x,(s,i)) (s, o) -> Diff m (x, (s, Vector i)) (s, Vector o)
scanlDiff layer = Diff run
  where run (x,(s,inputs)) = do (oks, s') <- runStateT (traverse (go_forward x) inputs) s
                                let
                                  (os, ks) = V.unzip oks
                                  back (ds', dos) = do
                                    (dxis, ds) <- runStateT (traverseBack go_backward (V.zip dos ks)) ds'
                                    let (dxs, dis) = V.unzip dxis
                                    dx <- plusm dxs
                                    return (dx, (ds, dis))
                                return ((s', os), back)

        -- go_forward :: x -> i -> StateT s m (o, _)
        go_forward x i = do s <- get
                            ((s',o), k) <- lift $ runDiff layer (x,(s,i))
                            put s'
                            return (o, k)

        -- go_backward :: (D o, _) -> StateT (D s) m (D x, D i)
        go_backward (dout, k) = do ds' <- get
                                   (dx, (ds, di)) <- lift $ k (ds', dout)
                                   put ds
                                   return (dx, di)

zipDiff :: (Applicative m) => Diff m (Vector x, Vector y) (Vector (x,y))
zipDiff = Diff run
  where
    run (xs, ys) = pure (V.zip xs ys, pure . V.unzip)

unzipDiff :: (Applicative m) => Diff m (Vector (x,y)) (Vector x, Vector y)
unzipDiff = Diff run
  where
    run (xys) = pure (V.unzip xys, pure . uncurry V.zip)

-- mapDiff :: forall m x i o. (Monad m, Additive m (D x)) =>
--              Diff m (x,i) o -> Diff m (x, Vector i) (Vector o)
-- mapDiff layer = Diff run
--   where run (x,inputs) = do oks <- traverse (go_forward x) inputs
--                             let
--                               (os, ks) = V.unzip oks
--                               back dos = do
--                                 dxis <- traverseBack go_backward (V.zip dos ks)
--                                 let (dxs, dis) = V.unzip dxis
--                                 dx <- plusm dxs
--                                 return (dx, dis)
--                             return (os, back)
--         go_forward x i = runDiff layer (x,i)
--         go_backward (dout, k) = k dout

mapDiff :: forall m i o. (Monad m) => Diff m i o -> Diff m (Vector i) (Vector o)
mapDiff layer = Diff run
  where run inputs = do oks <- traverse (runDiff layer) inputs
                        let
                          (os, ks) = V.unzip oks
                          back dos = traverse go_backward (V.zip dos ks)
                        return (os, back)
        go_backward (dout, k) = k dout

vsumDiff :: (Monad m, Additive m a) => Diff m (Vector a) a
vsumDiff = Diff run
  where
    run inputs = do out <- plusm inputs
                    let !n = V.length inputs
                        back dout = return (V.replicate n dout)
                    return (out, back)
