<!-- GENERATED FILE — do not edit by hand.
     Source: data/taxonomy.json
     Regenerate: python -m src.etl.render_taxonomy -->

# Label taxonomy

Two label levels are in play, and they are not the same thing.

**Fine intent (77 classes).** The label that ships with BANKING77. This is what
the models are actually trained to predict, and it is what every accuracy and
F1 figure over "intents" refers to.

**Coarse category (9 classes).** A triage queue. This is *our* mapping, defined
in `data/taxonomy.json` and written by hand for this project — it is not part of
the dataset and has no external validation. A category prediction is produced by
mapping the predicted intent through the table at the bottom of this file, not
by a second model. Category metrics therefore inherit every intent error that
crosses a category boundary, and are unaffected by intent errors that stay
inside one.

Each category carries the operational metadata the routing layer needs: the
queue it lands in, the SLA the business promises, and a base priority. The
routing rules in `src/routing/rules.py` may raise the priority above the base
(never lower it) — see that module for the escalation conditions.

## Categories

| Category | Queue | SLA (min) | Base priority | Security critical | Intents |
| --- | --- | ---: | ---: | :---: | ---: |
| `identity_security` | `FRAUD_SECURITY` | 15 | P1 | yes | 11 |
| `payments_disputes` | `DISPUTES` | 120 | P2 | no | 9 |
| `transfers` | `PAYMENTS_OPS` | 120 | P2 | no | 10 |
| `cards_lifecycle` | `CARD_OPS` | 480 | P3 | no | 9 |
| `cards_usage` | `CARD_SUPPORT` | 240 | P3 | no | 7 |
| `cash_atm` | `CARD_SUPPORT` | 240 | P3 | no | 5 |
| `top_up` | `PAYMENTS_OPS` | 240 | P3 | no | 8 |
| `account_admin` | `GENERAL_SUPPORT` | 720 | P4 | no | 7 |
| `fees_fx` | `BILLING` | 480 | P4 | no | 11 |

**`identity_security`** — Lost, stolen or compromised instruments, blocked credentials, and identity or source-of-funds verification. Highest priority because delay increases loss.

**`payments_disputes`** — Unrecognised, duplicated, declined, pending or reverted payments and refund requests. Money is already in dispute.

**`transfers`** — Inbound and outbound transfers: timing, failure, cancellation, beneficiaries and balances that have not updated.

**`cards_lifecycle`** — Ordering, delivery, activation, expiry and replacement of physical and virtual cards.

**`cards_usage`** — A card the customer already holds is not behaving: acceptance, contactless, linking, wallets, virtual card limits.

**`cash_atm`** — ATM availability, swallowed cards, declined or pending withdrawals and incorrect cash dispensed.

**`top_up`** — Adding funds to the account by card, bank transfer, cash or cheque, including limits, failures and reversals.

**`account_admin`** — Eligibility, supported countries and currencies, personal detail changes and account closure.

**`fees_fx`** — Questions about charges applied and exchange rates used. Usually informational rather than remedial.


`security_critical` categories bypass the low-confidence hold: a record routed
there is escalated to a human immediately rather than parked in a review queue,
because the cost of a delayed fraud report is higher than the cost of a false
alarm.

## Intent to category

All 77 intents, grouped by the category they map to.

### `identity_security` (11 intents)

- `change_pin`
- `compromised_card`
- `lost_or_stolen_card`
- `lost_or_stolen_phone`
- `passcode_forgotten`
- `pin_blocked`
- `unable_to_verify_identity`
- `verify_my_identity`
- `verify_source_of_funds`
- `verify_top_up`
- `why_verify_identity`

### `payments_disputes` (9 intents)

- `Refund_not_showing_up`
- `card_payment_not_recognised`
- `cash_withdrawal_not_recognised`
- `declined_card_payment`
- `direct_debit_payment_not_recognised`
- `pending_card_payment`
- `request_refund`
- `reverted_card_payment?`
- `transaction_charged_twice`

### `transfers` (10 intents)

- `balance_not_updated_after_bank_transfer`
- `beneficiary_not_allowed`
- `cancel_transfer`
- `declined_transfer`
- `failed_transfer`
- `pending_transfer`
- `receiving_money`
- `transfer_into_account`
- `transfer_not_received_by_recipient`
- `transfer_timing`

### `cards_lifecycle` (9 intents)

- `activate_my_card`
- `card_about_to_expire`
- `card_arrival`
- `card_delivery_estimate`
- `get_disposable_virtual_card`
- `get_physical_card`
- `getting_spare_card`
- `getting_virtual_card`
- `order_physical_card`

### `cards_usage` (7 intents)

- `apple_pay_or_google_pay`
- `card_acceptance`
- `card_linking`
- `card_not_working`
- `contactless_not_working`
- `disposable_card_limits`
- `virtual_card_not_working`

### `cash_atm` (5 intents)

- `atm_support`
- `card_swallowed`
- `declined_cash_withdrawal`
- `pending_cash_withdrawal`
- `wrong_amount_of_cash_received`

### `top_up` (8 intents)

- `automatic_top_up`
- `balance_not_updated_after_cheque_or_cash_deposit`
- `pending_top_up`
- `top_up_by_cash_or_cheque`
- `top_up_failed`
- `top_up_limits`
- `top_up_reverted`
- `topping_up_by_card`

### `account_admin` (7 intents)

- `age_limit`
- `country_support`
- `edit_personal_details`
- `fiat_currency_support`
- `supported_cards_and_currencies`
- `terminate_account`
- `visa_or_mastercard`

### `fees_fx` (11 intents)

- `card_payment_fee_charged`
- `card_payment_wrong_exchange_rate`
- `cash_withdrawal_charge`
- `exchange_charge`
- `exchange_rate`
- `exchange_via_app`
- `extra_charge_on_statement`
- `top_up_by_bank_transfer_charge`
- `top_up_by_card_charge`
- `transfer_fee_charged`
- `wrong_exchange_rate_for_cash_withdrawal`

