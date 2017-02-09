{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
module AI.Funn.SomeNat where

import           Control.Applicative
import           Control.Category
import           Data.Foldable
import           Data.Function
import           Data.Monoid
import           Data.Proxy

import           Data.Constraint

import           GHC.TypeLits

withNat :: Integer -> (forall n. (KnownNat n) => Proxy n -> r) -> r
withNat n f = case someNatVal n of
               Just (SomeNat proxy) -> f proxy
               Nothing              -> error ("withNat: negative value (" ++ show n ++ ")")
