{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, GADTs, RankNTypes, ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances, DataKinds #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ConstraintKinds #-}
module AI.Funn.Diff.Pointed (Pointed, runPointed, pushDiff, Ref, (<--), (=<=), (-<=), (=<-), Unpack(..), PackRec(..)) where

import           Prelude hiding ((.), id)

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Category
import           Data.Foldable
import           Data.Function hiding ((.), id)
import           Data.Monoid

import           Data.Maybe
import           Data.Proxy

import           Control.Lens
import           Data.Functor.Identity

import           Data.Random
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S

import           Control.Monad.Free

import           Data.Map (Map)
import qualified Data.Map.Strict as Map
import           Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IntMap

import           Control.DeepSeq

import           Unsafe.Coerce

import           AI.Funn.Diff.Diff (Additive(..), Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Diff.Dynamic

type Affine m a = (Derivable a, Additive m (D a))

liftDiffMap :: (Monad m, Affine m a, Derivable b) => Ref s a -> Ref s b -> Diff m a b -> Diff m (DiffMap s) (DiffMap s)
liftDiffMap ra rb diff = Diff run
  where
    run values = do
      let a = readMap ra values
      (b, k) <- Diff.runDiff diff a
      let values' = writeMap rb (Just b) values
      return (values', backward k)

    backward k dvs = do
      da <- case readMapD rb dvs of
              Just db -> Just <$> k db
              Nothing -> pure Nothing
      addMapD ra da dvs

data Step m s = Step [Int] [Int] (Diff m (DiffMap s) (DiffMap s))

data PointedF m s r =
  forall a. NewRef (Ref s a -> r)
  | TellStep (Step m s) r
deriving instance Functor (PointedF m s)
type Pointed m s = Free (PointedF m s)

newRef :: Pointed m s (Ref s a)
newRef = liftF (NewRef id)

tellStep :: Step m s -> Pointed m s ()
tellStep step = liftF (TellStep step ())

pushDiff :: (Monad m, Affine m a, Derivable b)
         => Ref s a -> Diff m a b -> Pointed m s (Ref s b)
pushDiff ra diff = do
  rb <- newRef
  let
    diffv = liftDiffMap ra rb diff
    step = Step [getRef ra] [getRef rb] diffv
  tellStep step
  return rb

(<--) :: (Monad m, Affine m a, Derivable b)
      => Diff m a b -> Ref s a -> Pointed m s (Ref s b)
(<--) = flip pushDiff

(=<=) :: (Monad m, Affine m p, Derivable q, Unpack m s p a, Unpack m s q b)
      => Diff m p q -> a -> Pointed m s b
diff =<= a = pack a >>= flip pushDiff diff >>= unpack

(-<=) :: (Monad m, Affine m p, Derivable b, Unpack m s p a)
      => Diff m p b -> a -> Pointed m s (Ref s b)
diff -<= a = pack a >>= flip pushDiff diff


(=<-) :: (Monad m, Affine m a, Derivable q, Unpack m s q b)
      => Diff m a q -> Ref s a -> Pointed m s b
diff =<- a = pushDiff a diff >>= unpack

runPoint :: (Ref s a -> Pointed m s (Ref s b)) -> (Ref s a, Ref s b, [Step m s])
runPoint f = case go 1 (f ref_a) of
               (ref_b, steps) -> (ref_a, ref_b, steps)
  where
    ref_a = Ref 0
    go index (Pure r) = (r, [])
    go index (Free (NewRef k)) = go (index + 1) (k (Ref index))
    go index (Free (TellStep s k)) = let res = go index k
                                     in (fst res, s : snd res)

instance Monad m => Monoid (Step m s) where
  mempty = Step [] [] id
  mappend (Step as bs d1) (Step as' bs' d2) =
    Step (as ++ as') (bs ++ bs') (d1 >>> d2)

runPointed :: (Monad m, Derivable a, Derivable b) => (forall s. Ref s a -> Pointed m s (Ref s b)) -> Diff m a b
runPointed f = Diff run
  where
    (ra, rb, steps) = runPoint f
    Step _ _ diff = fold steps
    run a = do
      let vsa = writeMap ra (Just a) emptyMap
      (vsb, k) <- Diff.runDiff diff vsa
      let b = readMap rb vsb
      return (b, backward k)

    backward k db = do
      let dvsb = writeMapD rb (Just db) emptyMapD
      dvsa <- k dvsb
      let Just a = readMapD ra dvsa
      return a

data Pack m s r a where
  (:%:) :: (Affine m a, Affine m b) => Ref s a -> Ref s b -> Pack m s (a, b) (Ref s a, Ref s b)
  (:<:) :: (Affine m a, Affine m t) => Ref s a -> Pack m s t b -> Pack m s (a, t) (Ref s a, b)
  (:>:) :: (Affine m r, Affine m b) => Pack m s r a -> Ref s b -> Pack m s (r, b) (a, Ref s b)
  (:&:) :: (Affine m r, Affine m t) => Pack m s r a -> Pack m s t b -> Pack m s (r, t) (a, b)

paq :: Monad m => Pack m s r a -> Pointed m s (Ref s r)
paq (ra :%: rb) = pack2 (ra, rb)
paq (ra :<: p) = do x <- paq p
                    pack2 (ra, x)
paq (p :>: rb) = do x <- paq p
                    pack2 (x, rb)
paq (p :&: q) = do x <- paq p
                   y <- paq q
                   pack2 (x, y)

class Unpack m s a b | b -> s, a s -> b, b -> a where
  unpack :: Ref s a -> Pointed m s b
  pack :: b -> Pointed m s (Ref s a)

instance (Monad m, Affine m a, Affine m b) =>
         Unpack m s (a, b) (Ref s a, Ref s b) where
  unpack = unpack2
  pack = pack2

instance (Monad m, Affine m a, Affine m b, Affine m c) =>
         Unpack m s (a,b,c) (Ref s a, Ref s b, Ref s c) where
  unpack = unpack3
  pack = pack3

unpack2 :: (Monad m, Affine m a, Affine m b)
        => Ref s (a, b) -> Pointed m s (Ref s a, Ref s b)
unpack2 rab =
  do ra <- newRef
     rb <- newRef
     let
       run vs =
         let (a,b) = readMap rab vs
             vso = writeMap ra (Just a) $ writeMap rb (Just b) $ vs
         in return (vso, backward)

       backward dvso = do
         da <- readMapDA ra dvso
         db <- readMapDA rb dvso
         addMapD rab (Just (da, db)) dvso

       diff = Diff run
       step = Step [getRef rab] [getRef ra, getRef rb] diff
     tellStep step
     return (ra, rb)

unpack3 :: (Monad m, Affine m a, Affine m b, Affine m c)
        => Ref s (a,b,c) -> Pointed m s (Ref s a, Ref s b, Ref s c)
unpack3 rabc =
  do ra <- newRef
     rb <- newRef
     rc <- newRef
     let
       run vs =
         let (a,b,c) = readMap rabc vs
             vso = writeMap ra (Just a) $
                   writeMap rb (Just b) $
                   writeMap rc (Just c) $
                   vs
         in return (vso, backward)

       backward dvso = do
         da <- readMapDA ra dvso
         db <- readMapDA rb dvso
         dc <- readMapDA rc dvso
         addMapD rabc (Just (da,db,dc)) dvso

       diff = Diff run
       step = Step [getRef rabc] [getRef ra, getRef rb, getRef rc] diff
     tellStep step
     return (ra, rb, rc)


pack2 :: (Monad m, Affine m a, Affine m b)
        => (Ref s a, Ref s b) -> Pointed m s (Ref s (a, b))
pack2 (ra, rb) =
  do rab <- newRef
     let
       run vs =
         let a = readMap ra vs
             b = readMap rb vs
             vso = writeMap rab (Just (a,b)) vs
         in return (vso, backward)

       backward dvso =
         case readMapD rab dvso of
           Just (da, db) -> addMapD ra (Just da) =<< addMapD rb (Just db) dvso
           Nothing -> pure dvso

       diff = Diff run
       step = Step [getRef ra, getRef rb] [getRef rab] diff
     tellStep step
     return rab

pack3 :: (Monad m, Affine m a, Affine m b, Affine m c)
        => (Ref s a, Ref s b, Ref s c) -> Pointed m s (Ref s (a, b, c))
pack3 (ra, rb, rc) =
  do rabc <- newRef
     let
       run vs =
         let a = readMap ra vs
             b = readMap rb vs
             c = readMap rc vs
             vso = writeMap rabc (Just (a,b,c)) vs
         in return (vso, backward)

       backward dvso =
         case readMapD rabc dvso of
           Just (da,db,dc) -> addMapD ra (Just da) =<< addMapD rb (Just db) =<< addMapD rc (Just dc) dvso
           Nothing -> pure dvso

       diff = Diff run
       step = Step [getRef ra, getRef rb, getRef rc] [getRef rabc] diff
     tellStep step
     return rabc

class PackRec m s shape p u | u -> p shape s, shape s p -> u where
  unpackrec :: Ref s p -> Pointed m s u
  packrec :: u -> Pointed m s (Ref s p)

instance Applicative m => PackRec m s () a (Ref s a) where
  unpackrec = pure
  packrec = pure

instance (Monad m,
          Affine m p_1,
          Affine m p_2,
          PackRec m s shape_1 p_1 u_1,
          PackRec m s shape_2 p_2 u_2) =>
         PackRec m s (shape_1, shape_2) (p_1, p_2) (u_1, u_2) where
  unpackrec ref = do (a_1, a_2) <- unpack ref
                     u_1 <- unpackrec a_1
                     u_2 <- unpackrec a_2
                     return (u_1, u_2)
  packrec (r1, r2) = do p1 <- packrec r1
                        p2 <- packrec r2
                        pack (p1, p2)

instance (Monad m,
          Affine m p_1,
          Affine m p_2,
          Affine m p_3,
          PackRec m s shape_1 p_1 u_1,
          PackRec m s shape_2 p_2 u_2,
          PackRec m s shape_3 p_3 u_3) =>
         PackRec m s (shape_1, shape_2, shape_3) (p_1, p_2, p_3) (u_1, u_2, u_3) where
  unpackrec ref = do (a_1, a_2, a_3) <- unpack ref
                     u_1 <- unpackrec a_1
                     u_2 <- unpackrec a_2
                     u_3 <- unpackrec a_3
                     return (u_1, u_2, u_3)
  packrec (r1, r2, r3) = do p1 <- packrec r1
                            p2 <- packrec r2
                            p3 <- packrec r3
                            pack (p1, p2, p3)
