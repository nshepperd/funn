{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, GADTs, RankNTypes, ScopedTypeVariables #-}
module AI.Funn.Pointed where

import           Prelude hiding ((.), id)

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Category
import           Data.Foldable
import           Data.Function hiding ((.), id)
import           Data.Monoid

import           Data.Proxy

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

data Ref s a = Ref Int

data Source m where
  Single :: (da ~ D a, VectorSpace da) => Proxy da -> Int -> Network m a b -> Source m

type PData m = (Int, Map Int (Source m))
type Pointed s (m :: * -> *) a = State (PData m) a

newRef :: Pointed s m (Ref s a)
newRef = do (next, map) <- get
            put (next + 1, map)
            return (Ref next)

putTable :: Ref s b -> Source m -> Pointed s m ()
putTable (Ref j) ss = do (next, map) <- get
                         let map' = Map.insert j ss map
                         put (next, map')

feed :: (da ~ D a, VectorSpace da) => Ref s a -> Network m a b -> Pointed s m (Ref s b)
feed (Ref i) network = do
  j <- newRef
  putTable j (Single Proxy i network)
  return j

-- split :: (da ~ D a, VectorSpace da) => Ref s (a,b) -> Pointed s m (Ref s a, Ref s b)

data Dynamic where
  Dynamic :: a -> Dynamic

unsafe :: Dynamic -> b
unsafe (Dynamic a) = unsafeCoerce a

splitpars :: Int -> Parameters -> (Parameters, Parameters)
splitpars n (Parameters xs) = (Parameters (V.take n xs), Parameters (V.drop n xs))

traverseBack :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)
traverseBack f = forwards . traverse (Backwards . f)

runPointed :: forall m a b. (Monad m) => (forall s. Ref s a -> Pointed s m (Ref s b)) -> Network m a b
runPointed k = Network ev numpar initial
  where
    (Ref finalref, (_, netmap)) = runState (k (Ref 0)) (1, Map.empty)
    ev pars a = do
      let
        go :: (Int, Source m) -> StateT (Map Int (Dynamic, Dynamic), Double, Parameters) m ()
        go (j, Single Proxy i network) = do
          (current, cost, cpars) <- get
          let
            (mypars, rest) = splitpars (params network) cpars
            b = if i == 0 then Dynamic a else fst (current Map.! i)
          (c, add_cost, k) <- lift $ evaluate network mypars (unsafe b)
          let
            new = Map.insert j (Dynamic c, Dynamic k) current
          put (new, cost + add_cost, rest)

      (cmap, cost, _) <- execStateT (traverse_ go (Map.toAscList netmap)) (Map.singleton 0 (Dynamic a, Dynamic (\da -> return (da, []) :: m (D a, [Parameters]))), 0, pars)
      let backward dc = do

            let goback :: (Int, Source m) -> StateT (Map Int Dynamic) m [Parameters]
                goback (j, Single Proxy i (network :: Network m x y)) = do
                  derivs <- get
                  let
                    myderiv :: D y
                    myderiv = case Map.lookup j derivs of
                        Just dyn -> unsafe dyn
                        Nothing -> error "no derivative"

                    k :: D y -> m (D x, [Parameters])
                    k = case Map.lookup j cmap of
                        Just (_, dk) -> unsafe dk
                        Nothing -> error "missing continuation"

                  (dx, dpars) <- lift $ k myderiv

                  let
                    newdx = case Map.lookup i derivs of
                               Just dx2 -> Dynamic (dx ## unsafe dx2)
                               Nothing -> Dynamic dx
                    new = Map.insert i newdx derivs
                  put new
                  return dpars
            (parslist, derivs) <- runStateT (traverseBack goback (Map.toAscList netmap)) Map.empty
            return (unsafe (derivs Map.! 0), concat parslist)

      return (unsafe (fst (cmap Map.! finalref)), cost, backward)

    numpar = sum $ [params network | (Single _ _ network) <- Map.elems netmap]
    initial = fold <$> sequence [initialise network | (Single _ _ network) <- Map.elems netmap]
