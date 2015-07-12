{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
module AI.Funn.Network (
  Parameters(..),
  VectorSpace(..),
  Derivable(..),
  Network(..),
  runNetwork, runNetwork',
  left, right, (>>>)
  ) where

import           Control.Applicative
import           Control.Category
import           Data.Foldable
import           Data.Function
import           Data.Monoid

import           Data.Functor.Identity

import qualified Data.Binary as B
import           Data.Random
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S

import           Control.DeepSeq

import           AI.Funn.Common

newtype Parameters = Parameters { getParameters :: S.Vector Double } deriving (Show, Read)

instance NFData Parameters where
  rnf (Parameters v) = rnf v

instance B.Binary Parameters where
  put (Parameters v) = putVector putDouble v
  get = Parameters <$> getVector getDouble

instance Monoid Parameters where
  {-# INLINE mempty #-}
  mempty = Parameters mempty
  {-# INLINE mappend #-}
  mappend (Parameters x) (Parameters y) = Parameters (x `mappend` y)

-- NEURALDATA

class VectorSpace d where
  (##) :: d -> d -> d
  unit :: d

instance (VectorSpace a, VectorSpace b) => VectorSpace (a, b) where
  (a1, b1) ## (a2, b2) = (a1 ## a2, b1 ## b2)
  unit = (unit, unit)

instance VectorSpace () where
  () ## () = ()
  unit = ()


class Derivable a where
  type family D a :: *

instance (Derivable a, Derivable b) => Derivable (a, b) where
  type D (a, b) = (D a, D b)

instance Derivable () where
  type D () = ()

instance Derivable Int where
  type D Int = ()

-- NETWORK

data Network m a b = Network {
  evaluate :: Parameters -> a ->
              m (b, Double, D b -> m (D a, [Parameters])),
  params :: Int,
  initialise :: RVar Parameters
  }

runNetwork :: Network Identity a b -> Parameters -> a -> (b, Double)
runNetwork network params a = let Identity (b, c, _) = evaluate network params a
                              in (b, c)

runNetwork' :: Network Identity a () -> Parameters -> a -> (Double, D a, Parameters)
runNetwork' network params a = let Identity ((), c, k) = evaluate network params a
                                   Identity (da, dparams) = k ()
                               in (c, da, fold dparams)

left :: (Monad m) => Network m a b -> Network m (a,c) (b,c)
left net = Network ev (params net) (initialise net)
  where
    ev par (a, c) = do (b, cost, k) <- evaluate net par a
                       let backward (db, dc) = do (da, dpar) <- k db
                                                  return ((da,dc), dpar)
                       return ((b,c), cost, backward)

right :: (Monad m) => Network m a b -> Network m (c,a) (c,b)
right net = Network ev (params net) (initialise net)
  where
    ev par (c, a) = do (b, cost, k) <- evaluate net par a
                       let backward (dc, db) = do (da, dpar) <- k db
                                                  return ((dc,da), dpar)
                       return ((c,b), cost, backward)

connect :: (Monad m) => Network m a b -> Network m b c -> Network m a c
connect one two = Network ev (params one + params two) (liftA2 (<>) (initialise one) (initialise two))
  where ev (Parameters par) !a = do (!b, !cost1, !k1) <- evaluate one (Parameters $ V.take (params one) par) a
                                    (!c, !cost2, !k2) <- evaluate two (Parameters $ V.drop (params one) par) b
                                    let backward !dc = do (!db, dpar2) <- k2 dc
                                                          (!da, dpar1) <- k1 db
                                                          return (da, dpar1 <> dpar2)
                                    return (c, cost1 + cost2, backward)

net_empty :: (Monad m) => Network m a a
net_empty = Network ev 0 (return mempty)
  where
    ev _ a = return (a, 0, backward)
    backward b = return (b, [])

instance Monad m => Category (Network m) where
  id = net_empty
  (.) = flip connect
