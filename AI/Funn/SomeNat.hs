{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RankNTypes #-}
module AI.Funn.SomeNat where

import           Control.Applicative
import           Control.Category
import           Data.Foldable
import           Data.Function
import           Data.Monoid
import           Data.Proxy

import           Data.Constraint
import           Unsafe.Coerce

import           Data.Functor.Identity

import           Data.Random
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Storable as S

import           Control.DeepSeq

import           GHC.TypeLits

import           AI.Funn.Flat
import           AI.Funn.Network

data SBlob = SBlob (S.Vector Double)

instance Derivable SBlob where
  type D SBlob = SBlob

withNat :: Integer -> (forall n. (KnownNat n) => Proxy n -> r) -> r
withNat n f = case someNatVal n of
               Just (SomeNat proxy) -> f proxy
               Nothing              -> error ("withNat: negative value (" ++ show n ++ ")")

dict :: (KnownNat n) => Proxy n -> Dict (KnownNat n)
dict Proxy = Dict

infixl 7 %*
(%*) :: forall a b. Dict (KnownNat a) -> Dict (KnownNat b) -> Dict (KnownNat (a*b))
Dict %* Dict = case someNatVal (natVal (Proxy :: Proxy a) * natVal (Proxy :: Proxy b)) of
                Just (SomeNat p) -> unsafeCoerce (dict p)

weakenL :: forall m c. (Monad m) => Integer -> (forall n. (KnownNat n) => Network m (Blob n) c) -> Network m SBlob c
weakenL n network = case someNatVal n of
                     Just (SomeNat proxy) -> sub proxy network
  where
    sub :: forall m n c. (Monad m, KnownNat n) => Proxy n -> Network m (Blob n) c -> Network m SBlob c
    sub Proxy (Network ev p i) = Network ev' p i
      where
        ev' pars (SBlob xs) = do (c, cost, k) <- ev pars (Blob xs)
                                 let backward dc = do (Blob da, dpars) <- k dc
                                                      return (SBlob da, dpars)
                                 return (c, cost, backward)

weakenR :: (Monad m) => Integer -> (forall n. (KnownNat n) => Network m a (Blob n)) -> Network m a SBlob
weakenR n network = case someNatVal n of
                     Just (SomeNat proxy) -> sub proxy network
  where
    sub :: forall m n a. (Monad m, KnownNat n) => Proxy n -> Network m a (Blob n) -> Network m a SBlob
    sub Proxy (Network ev p i) = Network ev' p i
      where
        ev' pars a = do (Blob cs, cost, k) <- ev pars a
                        let backward (SBlob dc) = k (Blob dc)
                        return (SBlob cs, cost, backward)
