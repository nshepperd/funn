{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, MultiParamTypeClasses, FlexibleInstances #-}
module AI.Funn.Network.Network (
  Parameters(..),
  Derivable(..),
  Additive(..),
  Diff(..),
  Network(..),
  runNetwork, runNetwork', runNetwork_,
  liftDiff,
  left, right, (>>>),
  (***), idWith,
  assocL, assocR, swap
  ) where

import           Prelude hiding ((.), id)

import           Control.Applicative
import           Control.Monad
import           Control.Category
import           Data.Foldable
import           Data.Monoid

import           Control.DeepSeq
import qualified Data.Binary as B
import           Data.Functor.Identity
import           Data.Random
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S

import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Additive(..), Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff

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

instance CheckNAN Parameters where
  check s (Parameters xs) b = if V.any (\x -> isNaN x || isInfinite x) xs then
                                error ("[" ++ s ++ "] checkNaN -- " ++ show b)
                              else ()

instance Derivable Parameters where
  type D Parameters = [Parameters]

data Network m a b = Network {
  evaluate :: Diff m (a, Parameters) (b, Double),
  params :: Int,
  initialise :: RVar Parameters
  }

runNetwork_ :: Network Identity a b -> Parameters -> a -> b
runNetwork_ network params a = let Identity ((b, _), _) = runDiff (evaluate network) (a, params) in b


runNetwork :: Network Identity a b -> Parameters -> a -> (b, Double)
runNetwork network params a = let Identity ((b, cost), _) = runDiff (evaluate network) (a, params)
                              in (b, cost)

runNetwork' :: Network Identity a () -> Parameters -> a -> (Double, D a, Parameters)
runNetwork' network params a = let (((), c), k) = runIdentity $ runDiff (evaluate network) (a, params)
                                   (da, dparams) = runIdentity $ k ((), 1)
                               in (c, da, fold dparams)

left :: (Monad m) => Network m a b -> Network m (a,c) (b,c)
left net = Network ev (params net) (initialise net)
  where
    ev = Diff.first Diff.swap >>> Diff.assocR >>> Diff.second (evaluate net) >>> Diff.assocL >>> Diff.first Diff.swap

right :: (Monad m) => Network m a b -> Network m (c,a) (c,b)
right net = Network ev (params net) (initialise net)
  where
    ev = Diff.assocR >>> Diff.second (evaluate net) >>> Diff.assocL

liftDiff :: (Monad m) => Diff m a b -> Network m a b
liftDiff diff = Network ev 0 (pure mempty)
  where
    ev = Diff.first diff >>> Diff.second (Diff (\_ -> return (0, \_ -> return [])))

assocL :: (Monad m) => Network m (a,(b,c)) ((a,b),c)
assocL = liftDiff Diff.assocL

assocR :: (Monad m) => Network m ((a,b),c) (a,(b,c))
assocR = liftDiff Diff.assocR

swap :: (Monad m) => Network m (a,b) (b,a)
swap = liftDiff Diff.swap

connect :: (Monad m) => Network m a b -> Network m b c -> Network m a c
connect one two = Network (Diff ev) (params one + params two) (liftA2 (<>) (initialise one) (initialise two))
  where ev (a, Parameters par) = do ((b, cost1), k1) <- runDiff (evaluate one) (a, Parameters $ V.take (params one) par)
                                    ((c, cost2), k2) <- runDiff (evaluate two) (b, Parameters $ V.drop (params one) par)
                                    let backward (dc, dcost) = do (db, dpar2) <- k2 (dc, dcost)
                                                                  (da, dpar1) <- k1 (db, dcost)
                                                                  return (da, dpar1 <> dpar2)
                                    return ((c, cost1 + cost2), backward)

net_empty :: (Monad m) => Network m a a
net_empty = liftDiff id

idWith :: (Monad m) => proxy a -> Network m a a
idWith _ = net_empty

instance Monad m => Category (Network m) where
  id = net_empty
  (.) = flip connect

infixr 3 ***
one *** two = left one >>> right two
