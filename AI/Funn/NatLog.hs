{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
module AI.Funn.NatLog (LogFloor, LogCeil, Max, witnessMax, witnessLogCeil, witness2) where

import Data.Constraint
import Data.Ord
import Data.Proxy
import GHC.TypeLits

import AI.Funn.SomeNat

type family RunOrdering (o :: Ordering) (a :: k) (b :: k) (c :: k) :: k where
  RunOrdering LT a _ _ = a
  RunOrdering EQ _ b _ = b
  RunOrdering GT _ _ c = c

-- type family IF (q :: Bool) (a :: k) (b :: k) :: k where
--   IF True a _ = a
--   IF False _ b = b

witnessMax :: forall proxy a b r. (KnownNat a, KnownNat b) =>
              proxy a -> proxy b -> (KnownNat (Max a b) => Proxy (Max a b) -> r) -> r
witnessMax pa pb r = withNatUnsafe (max a b) r
  where
    a = natVal pa
    b = natVal pb

witness2 :: forall proxy n r. (KnownNat n) =>
              proxy n -> (KnownNat (2^n) => Proxy (2^n) -> r) -> r
witness2 p r = withNatUnsafe (2^n) r
  where
    n = natVal p

witnessLogCeil :: forall proxy n r. (KnownNat n) =>
              proxy n -> (KnownNat (LogCeil n) => Proxy (LogCeil n) -> r) -> r
witnessLogCeil p r = withNatUnsafe d r
  where
    n = natVal p
    d = logCeil 32 n

    logCeil 0 n = 0
    logCeil s n = case compare (2^s) n of
                   LT -> s + 1
                   EQ -> s
                   GT -> logCeil (s - 1) n

type family Max (a :: Nat) (b :: Nat) :: Nat where
  Max 0 b = b
  Max a 0 = a
  Max a b = RunOrdering (CmpNat a b)
            b a a

type family LogFloor (n :: Nat) :: Nat where
  LogFloor 0 = 0
  LogFloor n = LogFloor32 n

type LogFloor1 n = RunOrdering (CmpNat (2^1) n) 1 1 0
type LogFloor2 n = RunOrdering (CmpNat (2^2) n) 2 2 (LogFloor1 n)
type LogFloor3 n = RunOrdering (CmpNat (2^3) n) 3 3 (LogFloor2 n)
type LogFloor4 n = RunOrdering (CmpNat (2^4) n) 4 4 (LogFloor3 n)
type LogFloor5 n = RunOrdering (CmpNat (2^5) n) 5 5 (LogFloor4 n)
type LogFloor6 n = RunOrdering (CmpNat (2^6) n) 6 6 (LogFloor5 n)
type LogFloor7 n = RunOrdering (CmpNat (2^7) n) 7 7 (LogFloor6 n)
type LogFloor8 n = RunOrdering (CmpNat (2^8) n) 8 8 (LogFloor7 n)
type LogFloor9 n = RunOrdering (CmpNat (2^9) n) 9 9 (LogFloor8 n)
type LogFloor10 n = RunOrdering (CmpNat (2^10) n) 10 10 (LogFloor9 n)
type LogFloor11 n = RunOrdering (CmpNat (2^11) n) 11 11 (LogFloor10 n)
type LogFloor12 n = RunOrdering (CmpNat (2^12) n) 12 12 (LogFloor11 n)
type LogFloor13 n = RunOrdering (CmpNat (2^13) n) 13 13 (LogFloor12 n)
type LogFloor14 n = RunOrdering (CmpNat (2^14) n) 14 14 (LogFloor13 n)
type LogFloor15 n = RunOrdering (CmpNat (2^15) n) 15 15 (LogFloor14 n)
type LogFloor16 n = RunOrdering (CmpNat (2^16) n) 16 16 (LogFloor15 n)
type LogFloor17 n = RunOrdering (CmpNat (2^17) n) 17 17 (LogFloor16 n)
type LogFloor18 n = RunOrdering (CmpNat (2^18) n) 18 18 (LogFloor17 n)
type LogFloor19 n = RunOrdering (CmpNat (2^19) n) 19 19 (LogFloor18 n)
type LogFloor20 n = RunOrdering (CmpNat (2^20) n) 20 20 (LogFloor19 n)
type LogFloor21 n = RunOrdering (CmpNat (2^21) n) 21 21 (LogFloor20 n)
type LogFloor22 n = RunOrdering (CmpNat (2^22) n) 22 22 (LogFloor21 n)
type LogFloor23 n = RunOrdering (CmpNat (2^23) n) 23 23 (LogFloor22 n)
type LogFloor24 n = RunOrdering (CmpNat (2^24) n) 24 24 (LogFloor23 n)
type LogFloor25 n = RunOrdering (CmpNat (2^25) n) 25 25 (LogFloor24 n)
type LogFloor26 n = RunOrdering (CmpNat (2^26) n) 26 26 (LogFloor25 n)
type LogFloor27 n = RunOrdering (CmpNat (2^27) n) 27 27 (LogFloor26 n)
type LogFloor28 n = RunOrdering (CmpNat (2^28) n) 28 28 (LogFloor27 n)
type LogFloor29 n = RunOrdering (CmpNat (2^29) n) 29 29 (LogFloor28 n)
type LogFloor30 n = RunOrdering (CmpNat (2^30) n) 30 30 (LogFloor29 n)
type LogFloor31 n = RunOrdering (CmpNat (2^31) n) 31 31 (LogFloor30 n)
type LogFloor32 n = RunOrdering (CmpNat (2^32) n) 32 32 (LogFloor31 n)

type family LogCeil (n :: Nat) :: Nat where
  LogCeil 0 = 0
  LogCeil n = LogCeil32 n

type LogCeil1 n = RunOrdering (CmpNat (2^1) n) 2 1 0
type LogCeil2 n = RunOrdering (CmpNat (2^2) n) 3 2 (LogCeil1 n)
type LogCeil3 n = RunOrdering (CmpNat (2^3) n) 4 3 (LogCeil2 n)
type LogCeil4 n = RunOrdering (CmpNat (2^4) n) 5 4 (LogCeil3 n)
type LogCeil5 n = RunOrdering (CmpNat (2^5) n) 6 5 (LogCeil4 n)
type LogCeil6 n = RunOrdering (CmpNat (2^6) n) 7 6 (LogCeil5 n)
type LogCeil7 n = RunOrdering (CmpNat (2^7) n) 8 7 (LogCeil6 n)
type LogCeil8 n = RunOrdering (CmpNat (2^8) n) 9 8 (LogCeil7 n)
type LogCeil9 n = RunOrdering (CmpNat (2^9) n) 10 9 (LogCeil8 n)
type LogCeil10 n = RunOrdering (CmpNat (2^10) n) 11 10 (LogCeil9 n)
type LogCeil11 n = RunOrdering (CmpNat (2^11) n) 12 11 (LogCeil10 n)
type LogCeil12 n = RunOrdering (CmpNat (2^12) n) 13 12 (LogCeil11 n)
type LogCeil13 n = RunOrdering (CmpNat (2^13) n) 14 13 (LogCeil12 n)
type LogCeil14 n = RunOrdering (CmpNat (2^14) n) 15 14 (LogCeil13 n)
type LogCeil15 n = RunOrdering (CmpNat (2^15) n) 16 15 (LogCeil14 n)
type LogCeil16 n = RunOrdering (CmpNat (2^16) n) 17 16 (LogCeil15 n)
type LogCeil17 n = RunOrdering (CmpNat (2^17) n) 18 17 (LogCeil16 n)
type LogCeil18 n = RunOrdering (CmpNat (2^18) n) 19 18 (LogCeil17 n)
type LogCeil19 n = RunOrdering (CmpNat (2^19) n) 20 19 (LogCeil18 n)
type LogCeil20 n = RunOrdering (CmpNat (2^20) n) 21 20 (LogCeil19 n)
type LogCeil21 n = RunOrdering (CmpNat (2^21) n) 22 21 (LogCeil20 n)
type LogCeil22 n = RunOrdering (CmpNat (2^22) n) 23 22 (LogCeil21 n)
type LogCeil23 n = RunOrdering (CmpNat (2^23) n) 24 23 (LogCeil22 n)
type LogCeil24 n = RunOrdering (CmpNat (2^24) n) 25 24 (LogCeil23 n)
type LogCeil25 n = RunOrdering (CmpNat (2^25) n) 26 25 (LogCeil24 n)
type LogCeil26 n = RunOrdering (CmpNat (2^26) n) 27 26 (LogCeil25 n)
type LogCeil27 n = RunOrdering (CmpNat (2^27) n) 28 27 (LogCeil26 n)
type LogCeil28 n = RunOrdering (CmpNat (2^28) n) 29 28 (LogCeil27 n)
type LogCeil29 n = RunOrdering (CmpNat (2^29) n) 30 29 (LogCeil28 n)
type LogCeil30 n = RunOrdering (CmpNat (2^30) n) 31 30 (LogCeil29 n)
type LogCeil31 n = RunOrdering (CmpNat (2^31) n) 32 31 (LogCeil30 n)
type LogCeil32 n = RunOrdering (CmpNat (2^32) n) 33 32 (LogCeil31 n)

-- p :: Int -> String
-- p s = "type LogFloor" ++ show s ++ " n = " ++
--       "RunOrdering (CmpNat (2^" ++ show s ++ ") n) " ++
--       show s ++ " " ++
--       show s ++ " " ++
--       "(LogFloor" ++ show (s-1) ++ " n)"

-- pc :: Int -> String
-- pc s = "type LogCeil" ++ show s ++ " n = " ++
--        "RunOrdering (CmpNat (2^" ++ show s ++ ") n) " ++
--        show (s+1) ++ " " ++
--        show s ++ " " ++
--        "(LogCeil" ++ show (s-1) ++ " n)"
