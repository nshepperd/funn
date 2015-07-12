{-# LANGUAGE TypeFamilies, KindSignatures, FlexibleContexts #-}
{-# LANGUAGE BangPatterns, ScopedTypeVariables, TypeOperators #-}
{-# LANGUAGE DataKinds #-}
module AI.Funn.WithParameters (
  VectorSpace(..),
  Derivable(..),
  Network(..),
  runNetwork, runNetwork',
  left, right, (>>>)
  ) where

import           Control.Applicative
import           Control.Monad
import           Data.Foldable
import           Data.Function
import           Data.Monoid
import           Data.Proxy

import           GHC.TypeLits
import           Data.Functor.Identity

import qualified Data.Binary as B
import           Data.Random
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Storable as S

import           Control.DeepSeq
import           AI.Funn.Common

class VectorSpace d where
  (##) :: d -> d -> d
  unit :: d

instance (VectorSpace a, VectorSpace b) => VectorSpace (a, b) where
  (a1, b1) ## (a2, b2) = (a1 ## a2, b1 ## b2)
  unit = (unit, unit)

instance VectorSpace () where
  () ## () = ()
  unit = ()


class (VectorSpace (D a)) => Derivable a where
  type family D a :: *

instance (Derivable a, Derivable b) => Derivable (a, b) where
  type D (a, b) = (D a, D b)

instance Derivable () where
  type D () = ()

instance Derivable Int where
  type D Int = ()

-- NETWORK

data Network m a b = Network {
  evaluate :: a -> m (b, Double, D b -> m (D a))
  }

runNetwork :: Network Identity a b -> a -> (b, Double)
runNetwork network a = let Identity (b, c, _) = evaluate network a
                       in (b, c)

runNetwork' :: Network Identity a () -> a -> (Double, D a)
runNetwork' network a = let Identity ((), c, k) = evaluate network a
                            Identity da = k ()
                        in (c, da)

left :: (Monad m) => Network m a b -> Network m (a,c) (b,c)
left net = Network ev
  where
    ev (a, c) = do (b, cost, k) <- evaluate net a
                   let backward (db, dc) = do da <- k db
                                              return (da,dc)
                   return ((b,c), cost, backward)

right :: (Monad m) => Network m a b -> Network m (c,a) (c,b)
right net = Network ev
  where
    ev (c, a) = do (b, cost, k) <- evaluate net a
                   let backward (dc, db) = do da <- k db
                                              return (dc,da)
                   return ((c,b), cost, backward)

connect :: (Monad m) => Network m a b -> Network m b c -> Network m a c
connect one two = Network ev
  where ev !a = do (!b, !cost1, !k1) <- evaluate one a
                   (!c, !cost2, !k2) <- evaluate two b
                   let backward = k2 >=> k1
                   return (c, cost1 + cost2, backward)

(>>>) :: (Monad m) => Network m a b -> Network m b c -> Network m a c
(>>>) = connect

newtype Blob (n :: Nat) = Blob { getBlob :: S.Vector Double }
                        deriving (Show)

natInt :: (KnownNat n) => proxy n -> Int
natInt = fromIntegral . natVal

zeroBlob :: forall n. (KnownNat n) => Blob n
zeroBlob = Blob (V.replicate (natInt (Proxy :: Proxy n)) 0)

instance (KnownNat n) => VectorSpace (Blob n) where
  (Blob x) ## (Blob y) = Blob (V.zipWith (+) x y)
  unit = zeroBlob

instance (KnownNat n) => Derivable (Blob n) where
  type D (Blob n) = Blob n


addBlob :: Blob n -> Blob m -> Blob (n+m)
addBlob (Blob x) (Blob y) = Blob (x <> y)

splitBlob :: forall a b. (KnownNat a, KnownNat b) => Blob (a + b) -> (Blob a, Blob b)
splitBlob (Blob xs) = (Blob (V.take s1 xs), Blob (V.drop s1 xs))
  where
    s1 = natInt (Proxy :: Proxy a)

connectP :: (KnownNat i, KnownNat j, Monad m) => Network m (Blob i, a) b -> Network m (Blob j, b) c -> Network m (Blob (i+j), a) c
connectP one two = Network ev
  where
    ev (pp, !a) = do let (p1, p2) = splitBlob pp
                     (!b, !cost1, !k1) <- evaluate one (p1, a)
                     (!c, !cost2, !k2) <- evaluate two (p2, b)
                     let backward dc = do (dp2, db) <- k2 dc
                                          (dp1, da) <- k1 db
                                          return (addBlob dp1 dp2, da)
                     return (c, cost1 + cost2, backward)
