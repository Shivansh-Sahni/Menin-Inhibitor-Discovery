#!/usr/bin/env perl
use strict;
use warnings;

# Stream BindingDB's full TSV on stdin. Output one physical-row table and one
# endpoint-measurement table for records whose Curation/DataSource is ChEMBL.
# The output is deliberately lossless for endpoint relation/value semantics.

my ($rows_path, $measurements_path) = @ARGV;
die "usage: $0 ROWS.tsv MEASUREMENTS.tsv\n" unless $rows_path && $measurements_path;

open my $rows_fh, '>', $rows_path or die "cannot write $rows_path: $!";
open my $meas_fh, '>', $measurements_path or die "cannot write $measurements_path: $!";

print {$rows_fh} join("\t", qw(
  bindingdb_reactant_set_id bindingdb_monomer_id chembl_ligand_id ligand_inchikey
  target_key target_key_status declared_chain_count resolved_accession_count
)), "\n";
print {$meas_fh} join("\t", qw(
  bindingdb_reactant_set_id bindingdb_monomer_id chembl_ligand_id ligand_inchikey
  target_key target_key_status endpoint relation value_nm value_raw
)), "\n";

my @endpoint_fields = (
  [8,  'Ki'],
  [9,  'IC50'],
  [10, 'Kd'],
  [11, 'EC50'],
);

my $header = <STDIN>;
die "empty input\n" unless defined $header;

while (my $line = <STDIN>) {
  chomp $line;
  $line =~ s/\r$//;
  my @f = split /\t/, $line, -1;
  next unless defined $f[16] && $f[16] eq 'ChEMBL';

  my $declared_chains = defined $f[39] ? $f[39] : '';
  my %accessions;
  for my $chain (0 .. 49) {
    my $swiss_idx = 44 + 12 * $chain;
    my $trembl_idx = 49 + 12 * $chain;
    last if $swiss_idx > $#f;
    my $raw = $f[$swiss_idx] ne '' ? $f[$swiss_idx] : ($trembl_idx <= $#f ? $f[$trembl_idx] : '');
    next if !defined($raw) || $raw eq '';
    for my $acc (split /[;,\s]+/, $raw) {
      $acc =~ s/^\s+|\s+$//g;
      next unless $acc =~ /^[A-Z0-9]+(?:-[0-9]+)?$/;
      $accessions{$acc} = 1;
    }
  }
  my @accessions = sort keys %accessions;
  my $target_key = join(';', @accessions);
  my $resolved = scalar @accessions;
  my $status = 'missing';
  if ($resolved > 0) {
    if ($declared_chains =~ /^\d+$/ && $declared_chains > 0 && $resolved >= $declared_chains) {
      $status = 'complete';
    } else {
      $status = 'partial_or_undeclared';
    }
  }

  my @base = map { defined($_) ? $_ : '' } (
    $f[0], $f[4], $f[34], $f[3], $target_key, $status,
  );
  print {$rows_fh} join("\t", @base, $declared_chains, $resolved), "\n";

  for my $spec (@endpoint_fields) {
    my ($idx, $endpoint) = @$spec;
    my $raw = defined $f[$idx] ? $f[$idx] : '';
    next if $raw =~ /^\s*$/;
    my ($relation, $numeric) = parse_affinity($raw);
    if (!defined $numeric) {
      $relation = 'UNPARSEABLE';
      $numeric = '';
    }
    print {$meas_fh} join("\t", @base, $endpoint, $relation, $numeric, sanitize($raw)), "\n";
  }
}

close $rows_fh or die "cannot close $rows_path: $!";
close $meas_fh or die "cannot close $measurements_path: $!";

sub parse_affinity {
  my ($raw) = @_;
  $raw =~ s/^\s+|\s+$//g;
  return unless $raw =~ /^(<=|>=|<<|>>|<|>|~|=)?\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$/;
  my $relation = defined($1) && $1 ne '' ? $1 : '=';
  my $number = 0 + $2;
  return ($relation, sprintf('%.15g', $number));
}

sub sanitize {
  my ($value) = @_;
  $value = '' unless defined $value;
  $value =~ s/[\t\r\n]+/ /g;
  return $value;
}
