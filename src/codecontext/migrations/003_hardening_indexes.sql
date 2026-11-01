CREATE INDEX IF NOT EXISTS relations_evidence ON relations(evidence_id);
CREATE INDEX IF NOT EXISTS coverage_gaps_record ON coverage_gaps(record_id);
CREATE INDEX IF NOT EXISTS coverage_gaps_scope_active ON coverage_gaps(scope, active);
