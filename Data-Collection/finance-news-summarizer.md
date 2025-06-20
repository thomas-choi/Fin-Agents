#if you use the OpenAI and Supabase, then need to follow
1.Create a .env file to put OPANAI_API_KEY and SUPABASE_KEY.
2.register a supabase account
3.create a null table with SQL, like below.
-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.finance_news (
  ticker text NOT NULL,
  date date NOT NULL,
  link text NOT NULL,
  title text NOT NULL,
  summary text NOT NULL,
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  current boolean NOT NULL DEFAULT true,
  CONSTRAINT finance_news_pkey PRIMARY KEY (id)
);
4.see SUPABASE_KEY from your setting -> API KEY
5.copy your SUPABASE_KEY from anon public, puting into your .env document.
6.go to API setting to copy your URL to replace the line 32.
7.disable the RLS.