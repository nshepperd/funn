{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.TypeLits (withNat, CLog, Max) where

import Data.Singletons
import Data.Constraint
import Data.Ord
import Data.Proxy
import Data.Singletons.TH (genDefunSymbols)
import GHC.TypeLits
import GHC.TypeLits.KnownNat
import Test.QuickCheck

withNat :: Integer -> (forall n. (KnownNat n) => Proxy n -> r) -> r
withNat n f = case someNatVal n of
               Just (SomeNat proxy) -> f proxy
               Nothing              -> error ("withNat: negative value (" ++ show n ++ ")")

type family RunOrdering (o :: Ordering) (a :: k) (b :: k) (c :: k) :: k where
  RunOrdering LT a _ _ = a
  RunOrdering EQ _ b _ = b
  RunOrdering GT _ _ c = c

-- Max: Maximum of two nats
-- instance (KnownNat a, KnownNat b) => KnownNat (Max a b)

type family Max (a :: Nat) (b :: Nat) :: Nat where
  Max 0 b = b
  Max a 0 = a
  Max a b = RunOrdering (CmpNat a b)
            b a a

-- MaxSym1 :: Nat -> (Nat ~> Nat)
data MaxSym1 (a :: Nat) (f :: TyFun Nat Nat)
type instance Apply (MaxSym1 a) b = Max a b
-- MaxSym0 :: Nat ~> (Nat ~> Nat)
data MaxSym0 (f :: TyFun Nat (Nat ~> Nat))
type instance Apply MaxSym0 a = MaxSym1 a

instance (KnownNat a, KnownNat b) => KnownNat2 "AI.Funn.TypeLits.Max" a b where
  type KnownNatF2 "AI.Funn.TypeLits.Max" = MaxSym0
  natSing2 = let x = natVal (Proxy :: Proxy a)
                 y = natVal (Proxy :: Proxy b)
                 z = max x y
             in SNatKn (fromIntegral z)

-- CLog x: exact integer equivalent of ceiling (logBase 2 x)
-- instance (KnownNat a) => KnownNat (CLog a)

type family CLog n where
  CLog 0 = 0
  CLog n = CLog32 n

type CLog1 n = RunOrdering (CmpNat (2^1) n) 2 1 0
type CLog2 n = RunOrdering (CmpNat (2^2) n) 3 2 (CLog1 n)
type CLog3 n = RunOrdering (CmpNat (2^3) n) 4 3 (CLog2 n)
type CLog4 n = RunOrdering (CmpNat (2^4) n) 5 4 (CLog3 n)
type CLog5 n = RunOrdering (CmpNat (2^5) n) 6 5 (CLog4 n)
type CLog6 n = RunOrdering (CmpNat (2^6) n) 7 6 (CLog5 n)
type CLog7 n = RunOrdering (CmpNat (2^7) n) 8 7 (CLog6 n)
type CLog8 n = RunOrdering (CmpNat (2^8) n) 9 8 (CLog7 n)
type CLog9 n = RunOrdering (CmpNat (2^9) n) 10 9 (CLog8 n)
type CLog10 n = RunOrdering (CmpNat (2^10) n) 11 10 (CLog9 n)
type CLog11 n = RunOrdering (CmpNat (2^11) n) 12 11 (CLog10 n)
type CLog12 n = RunOrdering (CmpNat (2^12) n) 13 12 (CLog11 n)
type CLog13 n = RunOrdering (CmpNat (2^13) n) 14 13 (CLog12 n)
type CLog14 n = RunOrdering (CmpNat (2^14) n) 15 14 (CLog13 n)
type CLog15 n = RunOrdering (CmpNat (2^15) n) 16 15 (CLog14 n)
type CLog16 n = RunOrdering (CmpNat (2^16) n) 17 16 (CLog15 n)
type CLog17 n = RunOrdering (CmpNat (2^17) n) 18 17 (CLog16 n)
type CLog18 n = RunOrdering (CmpNat (2^18) n) 19 18 (CLog17 n)
type CLog19 n = RunOrdering (CmpNat (2^19) n) 20 19 (CLog18 n)
type CLog20 n = RunOrdering (CmpNat (2^20) n) 21 20 (CLog19 n)
type CLog21 n = RunOrdering (CmpNat (2^21) n) 22 21 (CLog20 n)
type CLog22 n = RunOrdering (CmpNat (2^22) n) 23 22 (CLog21 n)
type CLog23 n = RunOrdering (CmpNat (2^23) n) 24 23 (CLog22 n)
type CLog24 n = RunOrdering (CmpNat (2^24) n) 25 24 (CLog23 n)
type CLog25 n = RunOrdering (CmpNat (2^25) n) 26 25 (CLog24 n)
type CLog26 n = RunOrdering (CmpNat (2^26) n) 27 26 (CLog25 n)
type CLog27 n = RunOrdering (CmpNat (2^27) n) 28 27 (CLog26 n)
type CLog28 n = RunOrdering (CmpNat (2^28) n) 29 28 (CLog27 n)
type CLog29 n = RunOrdering (CmpNat (2^29) n) 30 29 (CLog28 n)
type CLog30 n = RunOrdering (CmpNat (2^30) n) 31 30 (CLog29 n)
type CLog31 n = RunOrdering (CmpNat (2^31) n) 32 31 (CLog30 n)
type CLog32 n = RunOrdering (CmpNat (2^32) n) 33 32 (CLog31 n)

-- CLogSym0 :: (Nat ~> Nat)
data CLogSym0 (f :: TyFun Nat Nat)
type instance Apply CLogSym0 a = CLog a

cLog :: Integer -> Integer
cLog n = go 32
  where
    go 0 = 0
    go s = case compare (2^s) n of
             LT -> s + 1
             EQ -> s
             GT -> go (s - 1)

instance (KnownNat a) => KnownNat1 "AI.Funn.TypeLits.CLog" a where
  type KnownNatF1 "AI.Funn.TypeLits.CLog" = CLogSym0
  natSing1 = let x = natVal (Proxy :: Proxy a)
             in SNatKn (fromIntegral $ cLog x)

-- cLogTest :: Integer -> Integer
-- cLogTest n = withNat n $ \(Proxy :: Proxy n) ->
--                            natVal (Proxy :: Proxy (CLog n))

-- prop_cLogTypeLevel :: Property
-- prop_cLogTypeLevel = property $ \n -> n >= 0 ==> cLog n == cLogTest n

-- prop_cLogDefinition :: Property
-- prop_cLogDefinition = property $ \n -> n >= 0 ==> 2^(cLog n) >= n

-- prop_cLogDefinition' :: Property
-- prop_cLogDefinition' = property $ \n -> n >= 2 ==> 2^(cLog n - 1) < n
