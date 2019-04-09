{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module AI.Funn.Space where

import Control.Applicative
import Data.Foldable
import Data.Functor.Identity
import Data.Kind (Constraint)
import Data.Proxy
import Foreign.Storable
import GHC.Float
import GHC.Stack
import GHC.TypeLits

data Precision = FP32 | FP64
  deriving (Show, Eq, Ord)

class Storable a => Floats a where
  toDouble :: a -> Double
  fromDouble :: Double -> a
  precision :: Precision

convertFloat :: (Floats a, Floats b) => a -> b
convertFloat = fromDouble . toDouble

instance Floats Double where
  toDouble = id
  fromDouble = id
  precision = FP64

instance Floats Float where
  toDouble = float2Double
  fromDouble = double2Float
  precision = FP32

class Zero m a where
  zero :: m a

class Semi m a where
  plus :: a -> a -> m a

class Scale m x a | a -> x where
  scale :: x -> a -> m a

class (Zero m a, Semi m a) => Additive m a where
  plusm :: [a] -> m a
  default plusm :: (Monad m) => [a] -> m a
  plusm xs = do z <- zero
                foldrM plus z xs

class (Additive m a, Scale m x a) => VectorSpace m x a | a -> x where
  {}

class (VectorSpace m x a) => Inner m x a | a -> x where
  inner :: a -> a -> m x

class Inner m x a => Finite m x a | a -> x where
  getBasis :: a -> m [x]

(##) :: (Semi Identity a) => a -> a -> a
x ## y = runIdentity (plus x y)

unit :: (Zero Identity a) => a
unit = runIdentity zero


-- Trivial space: ()

instance (Applicative m) => Zero m () where
  zero = pure ()
instance (Applicative m) => Semi m () where
  plus _ _ = pure ()
instance (Applicative m) => Additive m () where
  plusm _ = pure ()
instance (Applicative m) => Scale m () () where
  scale _ _ = pure ()
instance (Applicative m) => VectorSpace m () ()
instance (Applicative m) => Inner m () () where
  inner _ _ = pure ()

instance (Applicative m) => Finite m () () where
  getBasis a = pure []

-- Almost trivial: floats

instance (Applicative m) => Zero m Double where
  zero = pure 0

instance (Applicative m) => Semi m Double where
  plus a b = pure (a + b)

instance (Applicative m) => Additive m Double where
  plusm xs = pure (sum xs)

instance (Applicative m) => Scale m Double Double where
  scale x a = pure (x * a)
instance (Applicative m) => VectorSpace m Double Double
instance (Applicative m) => Inner m Double Double where
  inner a b = pure (a * b)

instance (Applicative m) => Finite m Double Double where
  getBasis a = pure [a]


-- pairs

instance (Applicative m, Zero m a, Zero m b) => Zero m (a, b) where
  zero = liftA2 (,) zero zero

instance (Applicative m, Semi m a, Semi m b) => Semi m (a, b) where
  plus (a1, b1) (a2, b2) = liftA2 (,) (plus a1 a2) (plus b1 b2)

instance (Applicative m, Additive m a, Additive m b) => Additive m (a, b) where
  plusm abs = let (as, bs) = unzip abs
              in liftA2 (,) (plusm as) (plusm bs)

instance (Applicative m, Scale m x a, Scale m x b) => Scale m x (a,b) where
  scale x (a,b) = (,) <$> scale x a <*> scale x b

instance (Applicative m, VectorSpace m x a, VectorSpace m x b) => VectorSpace m x (a,b) where
  {}

instance (Applicative m, Num x, Inner m x a, Inner m x b) => Inner m x (a,b) where
  inner (a1, b1) (a2, b2) = (+) <$> inner a1 a2 <*> inner b1 b2

instance (Applicative m, Num x, Finite m x a, Finite m x b) => Finite m x (a,b) where
  getBasis (a, b) = (++) <$> getBasis a <*> getBasis b


-- triplets

instance (Applicative m, Zero m a, Zero m b, Zero m c) => Zero m (a, b, c) where
  zero = liftA3 (,,) zero zero zero

instance (Applicative m, Semi m a, Semi m b, Semi m c) => Semi m (a, b, c) where
  plus (a1, b1, c1) (a2, b2, c2) = liftA3 (,,) (plus a1 a2) (plus b1 b2) (plus c1 c2)

instance (Applicative m, Additive m a, Additive m b, Additive m c) => Additive m (a, b, c) where
  plusm abs = let (as, bs, cs) = unzip3 abs
              in liftA3 (,,) (plusm as) (plusm bs) (plusm cs)


instance (Applicative m, Scale m x a, Scale m x b, Scale m x c) => Scale m x (a,b,c) where
  scale x (a,b,c) = (,,) <$> scale x a <*> scale x b <*> scale x c

instance (Applicative m, VectorSpace m x a, VectorSpace m x b, VectorSpace m x c) => VectorSpace m x (a,b,c) where
  {}

instance (Applicative m, Num x, Inner m x a, Inner m x b, Inner m x c) => Inner m x (a,b,c) where
  inner (a1, b1, c1) (a2, b2, c2) = (\x y z -> x + y + z)
                                    <$> inner a1 a2
                                    <*> inner b1 b2
                                    <*> inner c1 c2

instance (Applicative m, Num x, Finite m x a, Finite m x b, Finite m x c) => Finite m x (a,b,c) where
  getBasis (a, b, c)  = (\x y z -> x ++ y ++ z)
                        <$> getBasis a
                        <*> getBasis b
                        <*> getBasis c

-- Dimensions --

data KnownListOf (ds :: [k]) where
  KnownListNil :: KnownListOf '[]
  KnownListCons :: KnownListOf ds -> KnownListOf (d ': ds)

class KnownList (ds :: [k]) where
  knownList :: proxy ds -> KnownListOf ds

instance KnownList '[] where
  knownList _ = KnownListNil

instance KnownList ds => KnownList (d ': ds) where
  knownList _ = KnownListCons (knownList Proxy)


type family Prod (ds :: [Nat]) :: Nat where
  Prod '[] = 1
  Prod (d ': ds) = d * Prod ds

type family KnownDimsF (ds :: [Nat]) :: Constraint where
  KnownDimsF '[] = ()
  KnownDimsF (d ': ds) = (KnownNat d, KnownDimsF ds)

data KnownDimsOf (ds :: [Nat]) where
  KnownDimsZero :: KnownDimsOf '[]
  KnownDimsSucc :: Int -> KnownDimsOf ds -> KnownDimsOf (d ': ds)

class (KnownNat (Prod ds), KnownList ds, KnownDimsF ds) => KnownDims (ds :: [Nat]) where
  knownDims :: proxy ds -> KnownDimsOf ds

instance KnownDims '[] where
  knownDims _ = KnownDimsZero

instance (KnownNat d, KnownDims ds) => KnownDims (d ': ds) where
  knownDims _ = KnownDimsSucc (fromIntegral (natVal (Proxy :: Proxy d))) (knownDims Proxy)

dimVal :: KnownDims ds => proxy ds -> [Int]
dimVal p = go (knownDims p)
  where
    go :: KnownDimsOf x -> [Int]
    go KnownDimsZero = []
    go (KnownDimsSucc d ds) = d : go ds

dimSize :: KnownDims ds => proxy ds -> Int
dimSize p = product (dimVal p)
