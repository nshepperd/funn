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

import           GHC.TypeLits

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


withNatUnsafe :: forall n r. Integer -> ((KnownNat n) => Proxy n -> r) -> r
withNatUnsafe n r = case someNatVal n of
                     Just (SomeNat proxy) -> case unsafeCoerce (dict proxy) :: Dict (KnownNat n) of
                                              Dict -> r (Proxy :: Proxy n)
                     Nothing              -> error ("withNat: negative value (" ++ show n ++ ")")
