{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, GADTs, RankNTypes, ScopedTypeVariables #-}
module AI.Funn.Pointed (runPointed, feed, Ref, joinP) where

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

import           Control.Monad.State.Lazy

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import           Control.DeepSeq

import           Unsafe.Coerce

import           AI.Funn.Common
import           AI.Funn.Network

newtype Ref s a = Ref { getRef :: Int }

data Source s m where
  Single :: (VectorSpace (D a), VectorSpace (D b)) => Ref s a -> Ref s b -> Network m a b -> Source s m
  Join :: (VectorSpace (D a), VectorSpace (D b)) => Ref s a -> Ref s b -> Ref s (a,b) -> Source s m

data Dynamic where
  Dynamic :: a -> Dynamic

data PData s m = PData { _pNextRef :: Int,
                         _pTable :: [Source s m] }
type Pointed s (m :: * -> *) a = State (PData s m) a

pNextRef :: Lens' (PData s m) Int
pNextRef f (PData a b) = (`PData` b) <$> f a

pTable :: Lens' (PData s m) [Source s m]
pTable f (PData a b) = PData a <$> f b

newRef :: Pointed s m (Ref s a)
newRef = do next <- use pNextRef
            pNextRef .= next+1
            return (Ref next)

feed :: (VectorSpace (D a), VectorSpace (D b)) => Ref s a -> Network m a b -> Pointed s m (Ref s b)
feed i network = do
  j <- newRef
  pTable %= (Single i j network :)
  return j

joinP :: (VectorSpace (D a), VectorSpace (D b)) => Ref s a -> Ref s b -> Pointed s m (Ref s (a,b))
joinP i j = do
  k <- newRef
  pTable %= (Join i j k :)
  return k

type Forward = Map Int Dynamic

fValue :: Ref s a -> Lens' Forward (Maybe a)
fValue (Ref i) = lens
                 (\m -> unsafe <$> Map.lookup i m)
                 (\m x -> case x of
                           Just z -> Map.insert i (Dynamic z) m
                           Nothing -> Map.delete i m)

type Backward = Map Int Dynamic

bValue :: Ref s a -> Lens' Backward (Maybe (D a))
bValue (Ref i) = lens
                 (\m -> unsafe <$> Map.lookup i m)
                 (\m x -> case x of
                           Just z -> Map.insert i (Dynamic z) m
                           Nothing -> Map.delete i m)

type K m a b = D b -> m (D a, [Parameters])

data Back s m where
  SingleB :: (VectorSpace (D a), VectorSpace (D b)) => Ref s a -> Ref s b -> K m a b -> Back s m
  JoinB :: (VectorSpace (D a), VectorSpace (D b)) => Ref s a -> Ref s b -> Ref s (a,b) -> Back s m

unsafe :: Dynamic -> b
unsafe (Dynamic a) = unsafeCoerce a

splitpars :: Int -> Parameters -> (Parameters, Parameters)
splitpars n (Parameters xs) = (Parameters (V.take n xs), Parameters (V.drop n xs))

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

runPointed :: forall m a b. (Monad m) => (forall s. Ref s a -> Pointed s m (Ref s b)) -> Network m a b
runPointed k = Network ev numpar initial
  where
    ref0 = Ref 0 :: Ref s a
    (finalref, PData _ sources') = runState (k (Ref 0)) (PData 1 [])
    sources = reverse sources'
    (indexes, numpar) = let go :: Source s m -> Int
                            go (Single i j network) = params network
                            go (Join i j k) = 0
                        in runState (traverse ((\x -> get <* modify (+x)) . go) sources) 0

    ev pars a = do
      let
        go :: (Source s m, Int) -> StateT Forward m (Double, Back s m)
        go (Single i j network, index) = do
          Just a <- use (fValue i)
          let mypars = Parameters (V.slice index (params network) (getParameters pars))
          (b, add_cost, k) <- lift $ evaluate network mypars a
          fValue j .= Just b
          return (add_cost, SingleB i j k)
        go (Join i j k, _) = do
          Just a <- use (fValue i)
          Just b <- use (fValue j)
          fValue k .= Just (a,b)
          return (0, JoinB i j k)

      (results, values) <- runStateT (traverse go (zip sources indexes)) (Map.singleton 0 (Dynamic a))
      let
        cost = sum (map fst results)
        back = map snd results

        backward db = do
          let goback :: Back s m -> StateT Backward m [Parameters]
              goback (SingleB i j k) = do
                may_db <- use (bValue j)
                let db = case may_db of
                      Just db -> db
                      Nothing -> unit
                (da, dpars) <- lift (k db)
                da_old <- use (bValue i)
                bValue i .= case da_old of
                  Just da1 -> Just (da ## da1)
                  Nothing -> Just da
                return dpars

              goback (JoinB i j k) = do
                may_dc <- use (bValue k)
                let (da,db) = case may_dc of
                      Just dc -> dc
                      Nothing -> unit

                da_old <- use (bValue i)
                bValue i .= case da_old of
                  Just da1 -> Just (da ## da1)
                  Nothing -> Just da

                db_old <- use (bValue j)
                bValue j .= case db_old of
                  Just db1 -> Just (db ## db1)
                  Nothing -> Just db

                return []

          (parslist, bmap) <- runStateT (traverseBack goback back) (Map.singleton (getRef finalref) (Dynamic db))
          let Just da = view (bValue ref0) bmap
          return (da, concat parslist)

      let Just b = view (fValue finalref) values
      return (b, cost, backward)

    initial = fold <$> sequence [initialise network | (Single _ _ network) <- sources]
